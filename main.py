import os
import re
import json
import asyncio
import logging
from html import escape as h
from collections import OrderedDict
from contextlib import asynccontextmanager

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
    LabeledPrice,
    PreCheckoutQuery,
    CallbackQuery,
)

from google import genai
from google.genai import types as genai_types

# =====================================================================
# SETTINGS
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
APP_URL = os.getenv("RENDER_EXTERNAL_URL", "https://sniper-ugc.onrender.com").strip()
APP_URL_VERSIONED = APP_URL.rstrip("/") + "/?v=8"

ADMIN_TG_ID = os.getenv("ADMIN_TG_ID", "").strip()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.0-flash").split(",")
    if m.strip()
]

REQUEST_TIMEOUT_MS = 45000
TOTAL_BUDGET_SEC = 95
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "1").strip() == "1"
MAX_OUTPUT_TOKENS = 4096

bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
dp = Dispatcher()


def make_ai_client():
    if not GEMINI_API_KEY:
        return None
    try:
        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        )
    except Exception as e:
        logging.warning(f"HttpOptions not supported, using default client: {e}")
        return genai.Client(api_key=GEMINI_API_KEY)


ai_client = make_ai_client()

# ---- Auto-discovery of available Gemini models ----
_MODEL_CACHE = {"ts": 0.0, "models": []}
_BAD_WORDS = (
    "image", "tts", "audio", "live", "embed", "vision", "robotics",
    "computer", "learnlm", "gemma", "aqa", "imagen", "veo"
)


def _model_version(name: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
    return float(m.group(1)) if m else 0.0


def discover_models_sync() -> list:
    found = []
    for m in ai_client.models.list():
        name = (getattr(m, "name", "") or "").replace("models/", "")
        actions = getattr(m, "supported_actions", None) or []
        if "flash" not in name or any(w in name for k in _BAD_WORDS for w in [k]):
            continue
        if actions and "generateContent" not in actions:
            continue
        found.append(name)
    found.sort(key=lambda n: (("preview" in n) or ("exp" in n), -_model_version(n), "lite" in n))
    return found


async def get_model_candidates() -> list:
    static = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    loop = asyncio.get_running_loop()
    now = loop.time()
    if not _MODEL_CACHE["models"] or now - _MODEL_CACHE["ts"] > 3600:
        try:
            models = await asyncio.wait_for(asyncio.to_thread(discover_models_sync), timeout=10)
            _MODEL_CACHE["models"] = models
            logging.info(f"Available flash models: {models[:6]}")
        except Exception as e:
            logging.warning(f"Model discovery failed: {e}")
        _MODEL_CACHE["ts"] = now
    discovered = _MODEL_CACHE["models"]
    if discovered:
        ordered = ([GEMINI_MODEL] if GEMINI_MODEL in discovered else []) + discovered
        return list(dict.fromkeys(ordered))[:4]
    return list(dict.fromkeys(static))


# =====================================================================
# REQUEST SCHEMAS & PACKAGES (100 STARS / SCRIPT, 75 STARS / POST)
# =====================================================================
class PreviewRequest(BaseModel):
    query: str


class AnalyzeRequest(BaseModel):
    query: str
    format_type: str = "ugc"
    tone: str = "native"
    lang: str = "ru"


class RewriteScriptRequest(BaseModel):
    product_title: str
    script: dict
    action: str = "shorter"   # shorter, longer, bolder, native, funny, custom
    custom_instruction: str = ""
    lang: str = "ru"


class TgPostRequest(BaseModel):
    query: str
    style: str = "native"
    lang: str = "ru"


class StarsInvoiceRequest(BaseModel):
    pack_id: str
    user_id: int = 0
    lang: str = "ru"


STAR_PACKS = {
    "pack_100": {
        "credits": 100,
        "stars": 100,
        "title_ru": "100 ★ SNIPER UGC (1 сценарий)",
        "title_en": "100 ★ SNIPER UGC (1 Script)",
        "desc_ru": "Пополнение на 100 ★ (1 полный сценарий с суфлёром или TG-пост с остатком)",
        "desc_en": "Top up 100 ★ (1 full script funnel or Telegram Ad Post)",
    },
    "pack_300": {
        "credits": 300,
        "stars": 300,
        "title_ru": "300 ★ SNIPER UGC (Хит)",
        "title_en": "300 ★ SNIPER UGC (Popular)",
        "desc_ru": "Пакет на 3 полных сценария (по 100 ★) или 4 рекламных TG-поста (по 75 ★)",
        "desc_en": "Pack for 3 full scripts (100 ★ each) or 4 Telegram Ad Posts (75 ★ each)",
    },
    "pack_1000": {
        "credits": 1000,
        "stars": 850,
        "title_ru": "1000 ★ SNIPER PRO (Выгода 15%)",
        "title_en": "1000 ★ SNIPER PRO (Save 15%)",
        "desc_ru": "PRO-пакет: 1000 ★ на баланс всего за 850 Telegram Stars",
        "desc_en": "PRO pack: 1000 ★ balance for 850 Telegram Stars",
    },
}


# =====================================================================
# ERRORS (localized)
# =====================================================================
class AppError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


ERR = {
    "ru": {
        "no_key": "GEMINI_API_KEY не задан на сервере (Render → Environment).",
        "empty": "Модель вернула пустой ответ. Попробуй ещё раз.",
        "bad_json": "Модель вернула некорректный ответ. Попробуй ещё раз.",
        "busy": "Сервера Gemini сейчас перегружены. Подожди пару секунд и нажми ещё раз.",
        "quota": "Превышен лимит запросов к Gemini. Попробуй чуть позже.",
        "key": "Проблема с ключом Gemini API. Проверь GEMINI_API_KEY.",
        "no_model": "Не нашлось доступной модели Gemini. Проверь GEMINI_MODEL в Environment.",
        "timeout": "Генерация заняла слишком много времени. Попробуй ещё раз.",
        "short_query": "Введи ссылку, артикул или описание товара.",
        "unknown": "Неизвестная ошибка",
    },
    "en": {
        "no_key": "GEMINI_API_KEY is not set on the server (Render → Environment).",
        "empty": "The model returned an empty response. Please try again.",
        "bad_json": "The model returned an invalid response. Please try again.",
        "busy": "Gemini servers are overloaded right now. Wait a few seconds and try again.",
        "quota": "Gemini request limit exceeded. Please try again a bit later.",
        "key": "Gemini API key problem. Check GEMINI_API_KEY.",
        "no_model": "No available Gemini model found. Check GEMINI_MODEL in Environment.",
        "timeout": "Generation took too long. Please try again.",
        "short_query": "Enter a link, SKU or product description.",
        "unknown": "Unknown error",
    },
}

TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "high demand", "overloaded", "429",
    "RESOURCE_EXHAUSTED", "500", "INTERNAL", "DEADLINE", "timed out",
    "timeout", "Timeout", "ReadError", "ConnectError", "RemoteProtocolError",
)
GONE_MARKERS = ("NOT_FOUND", "404", "no longer available")


def classify(e: Exception) -> str:
    if isinstance(e, AppError):
        return e.code
    m = str(e)
    if any(k in m for k in ("503", "UNAVAILABLE", "high demand", "overloaded")):
        return "busy"
    if "429" in m or "RESOURCE_EXHAUSTED" in m:
        return "quota"
    if any(k in m for k in ("API_KEY", "API key", "PERMISSION_DENIED", "UNAUTHENTICATED")):
        return "key"
    if any(k in m for k in GONE_MARKERS):
        return "no_model"
    if any(k in m for k in ("timed out", "timeout", "Timeout", "DEADLINE")):
        return "timeout"
    return "unknown"


def error_text(e: Exception, lang: str) -> str:
    lang = lang if lang in ERR else "en"
    code = classify(e)
    text = ERR[lang].get(code, ERR[lang]["unknown"])
    if code == "unknown":
        text += f": {str(e)[:160]}"
    return text


def norm_lang(code: str) -> str:
    return "ru" if (code or "")[:2].lower() in ("ru", "uk", "be", "kk") else "en"


# =====================================================================
# DATA COLLECTOR (WILDBERRIES / LINKS / TEXT)
# =====================================================================
_PREVIEW_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def get_wb_basket_host(vol: int) -> str:
    ranges = [
        (143, "01"), (287, "02"), (431, "03"), (719, "04"), (1007, "05"),
        (1061, "06"), (1115, "07"), (1169, "08"), (1313, "09"), (1601, "10"),
        (1655, "11"), (1919, "12"), (2045, "13"), (2189, "14"), (2405, "15"),
        (2621, "16"), (2837, "17"), (3053, "18"), (3269, "19"), (3485, "20"),
        (3701, "21"), (3917, "22"), (4133, "23"), (4349, "24")
    ]
    for limit, num in ranges:
        if vol <= limit:
            return f"https://basket-{num}.wbbasket.ru"
    return "https://basket-25.wbbasket.ru"


async def find_wb_card(vol: int, part: int, nm_id: int, client: httpx.AsyncClient):
    path = f"/vol{vol}/part{part}/{nm_id}"
    default_base = get_wb_basket_host(vol) + path

    try:
        r = await client.get(f"{default_base}/info/ru/card.json", timeout=4.5)
        if r.status_code == 200:
            return default_base, r.json()
    except Exception as e:
        logging.warning(f"WB primary basket failed: {e}")

    async def probe(num: int):
        base = f"https://basket-{num:02d}.wbbasket.ru{path}"
        try:
            r = await client.get(f"{base}/info/ru/card.json", timeout=3.5)
            if r.status_code == 200:
                return base, r.json()
        except Exception:
            pass
        return None

    results = await asyncio.gather(*(probe(n) for n in range(1, 41)))
    for res in results:
        if res:
            return res
    return default_base, None


async def parse_wildberries_safe(nm_id: int, client: httpx.AsyncClient) -> dict:
    vol = nm_id // 100000
    part = nm_id // 1000

    base_url, card = await find_wb_card(vol, part, nm_id, client)

    info = {
        "source": "Wildberries",
        "id": str(nm_id),
        "title": f"Wildberries #{nm_id}",
        "brand": "",
        "category": "",
        "description": "",
        "features": [],
        "price": "",
        "image": f"{base_url}/images/big/1.webp"
    }

    if card:
        try:
            info["title"] = card.get("imt_name") or info["title"]
            info["brand"] = (card.get("selling") or {}).get("brand_name", "")
            info["category"] = card.get("subj_name") or card.get("subj_root_name") or ""
            info["description"] = (card.get("description") or "")[:1500]
            opts = card.get("options") or []
            info["features"] = [f"{o.get('name')}: {o.get('value')}" for o in opts[:8]]
        except Exception as e:
            logging.warning(f"WB card parse warning: {e}")

    try:
        card_api = f"https://card.wb.ru/cards/v2/detail?appType=1&curr=rub&dest=-1257786&spp=30&nm={nm_id}"
        r_detail = await client.get(card_api, timeout=5.0)
        if r_detail.status_code == 200:
            prods = r_detail.json().get("data", {}).get("products", [])
            if prods:
                p = prods[0]
                if not info["brand"]:
                    info["brand"] = p.get("brand", "")
                if info["title"].startswith("Wildberries #"):
                    info["title"] = p.get("name", info["title"])
                sizes = p.get("sizes", [])
                if sizes and "price" in sizes[0]:
                    pr = sizes[0]["price"].get("product", 0) // 100
                    if pr:
                        info["price"] = f"{pr} ₽"
    except Exception as e:
        logging.warning(f"WB detail API warning: {e}")

    return info


async def extract_product_data(raw_input: str) -> dict:
    raw = raw_input.strip()
    cache_key = raw.lower()
    if cache_key in _PREVIEW_CACHE:
        _PREVIEW_CACHE.move_to_end(cache_key)
        return _PREVIEW_CACHE[cache_key]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8"
    }

    wb_match = re.search(r"wildberries\.ru/catalog/(\d+)", raw)
    num_match = re.fullmatch(r"\d{6,12}", raw)
    if wb_match or num_match:
        nm_id = int(wb_match.group(1) if wb_match else raw)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            res = await parse_wildberries_safe(nm_id, client)
            _PREVIEW_CACHE[cache_key] = res
            if len(_PREVIEW_CACHE) > 80:
                _PREVIEW_CACHE.popitem(last=False)
            return res

    url_match = re.search(r"(https?://[^\s]+)", raw)
    if url_match:
        url = url_match.group(1)
        domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        source_label = "Ozon" if "ozon." in domain else ("Yandex Market" if "market.yandex" in domain else domain[:22])
        fallback_data = {
            "source": source_label,
            "id": "",
            "url": url,
            "title": f"Товар ({source_label})",
            "brand": "",
            "category": "",
            "price": "",
            "description": "",
            "features": [],
            "image": "",
            "raw_text": f"Product link: {url}\nUser context: {raw}"
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=7.5) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    og_title = soup.find("meta", property="og:title")
                    og_desc = soup.find("meta", property="og:description")
                    og_img = soup.find("meta", property="og:image")

                    if og_title and og_title.get("content"):
                        fallback_data["title"] = og_title["content"].strip()
                    elif soup.title and soup.title.string:
                        fallback_data["title"] = soup.title.string.strip()

                    if og_desc and og_desc.get("content"):
                        fallback_data["description"] = og_desc["content"].strip()
                    if og_img and og_img.get("content"):
                        fallback_data["image"] = og_img["content"].strip()

                    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
                        tag.decompose()
                    clean = " ".join(soup.stripped_strings)
                    fallback_data["raw_text"] += f"\nSite content: {clean[:1800]}"
        except Exception as e:
            logging.warning(f"URL parsing skipped due to: {e}")

        _PREVIEW_CACHE[cache_key] = fallback_data
        if len(_PREVIEW_CACHE) > 80:
            _PREVIEW_CACHE.popitem(last=False)
        return fallback_data

    return {
        "source": "Описание",
        "id": "",
        "title": raw[:72],
        "brand": "",
        "category": "",
        "price": "",
        "description": raw[:1500],
        "features": [],
        "image": ""
    }


# =====================================================================
# GEMINI GENERATION (SCRIPTS + REWRITE + TG AD POSTS)
# =====================================================================
SYSTEM_INSTRUCTION = """
You are an elite creative producer of viral short-form content for TikTok, Reels, Shorts, and Telegram Ad Posts.
Analyze the product and produce a high-converting promotion funnel.
You ALWAYS reply with STRICTLY valid JSON, no markdown, no ```json fences.

JSON SCHEMA:
{
  "product_title": "string",
  "category": "string",
  "target_audience": "string (one sentence)",
  "pain_points": ["string", "string", "string"],
  "killer_features": ["string", "string", "string"],
  "viral_hooks": ["string", "string", "string", "string", "string"],
  "scripts": [
    {
      "title": "string",
      "angle": "string",
      "virality_score": 96,
      "hook": "string",
      "scenes": [
        {
          "timecode": "0:00-0:03",
          "visual": "string",
          "voiceover": "string (natural spoken phrase for teleprompter, max 18 words)",
          "on_screen_text": "string (max 6 words)",
          "ai_video_prompt": "English cinematic prompt for Kling/Runway, max 28 words"
        }
      ],
      "cta": "string"
    }
  ],
  "tg_posts": [
    {
      "style": "Нативная рекомендация",
      "headline": "Цепляющий заголовок поста с эмодзи",
      "body": "Живой текст поста для Telegram (разделенный на 2-3 коротких абзаца, с буллитами преимуществ, без воды и штампов)",
      "cta": "Где купить / призыв перейти по ссылке или артикулу"
    }
  ],
  "video_generator_prompts": ["English prompt", "English prompt"],
  "seo_caption": "string (2 sentences max)",
  "hashtags": ["#tag", "#tag", "#tag", "#tag", "#tag"]
}
"""

REWRITE_SYSTEM_INSTRUCTION = """
You are an elite TikTok/Reels script doctor.
Rewrite the provided script according to the user's modification command while keeping the same JSON structure.
Reply STRICTLY with valid JSON representing the single updated script object, no markdown fences.

JSON SCHEMA:
{
  "script": {
    "title": "string",
    "angle": "string",
    "virality_score": 97,
    "hook": "string",
    "scenes": [
      {
        "timecode": "0:00-0:02",
        "visual": "string",
        "voiceover": "string",
        "on_screen_text": "string",
        "ai_video_prompt": "English cinematic prompt"
      }
    ],
    "cta": "string"
  }
}
"""

TG_POST_SYSTEM_INSTRUCTION = """
You are an expert copywriter for Telegram channels and bloggers doing native integrations and marketplace (Wildberries/Ozon) product seeding.
Generate 3 distinct, high-converting Telegram posts for the product.
Reply STRICTLY with valid JSON, no markdown fences.

JSON SCHEMA:
{
  "product_title": "string",
  "tg_posts": [
    {
      "style": "string (e.g. Личная рекомендация / Находка дня / Горячий оффер)",
      "headline": "string (Punchy headline with emoji)",
      "body": "string (Engaging Telegram post text, 3-5 short lines, bullet points with emojis, conversational tone)",
      "cta": "string (Call to action with placeholder for link/SKU)"
    }
  ]
}
"""

FORMAT_LABELS = {
    "ugc": "Native honest review from a real user",
    "problem_solution": "Sharp audience pain -> rescue with the product",
    "unboxing": "Quick unboxing / ASMR / first impression",
    "aesthetic": "Aesthetic / TikTok lifestyle / minimal visuals",
    "storytelling": "Life story / provocation / unexpected ending",
}
TONE_LABELS = {
    "native": "Conversational, native, trustworthy",
    "aggressive": "Bold, clickbait, fast pace, hard triggers",
    "humor": "Irony, light jokes, everyday humor",
    "luxury": "Premium, calm, focus on status and quality",
}
REWRITE_ACTIONS = {
    "shorter": "Make the script 40% shorter, faster-paced, ultra-punchy, cut all fluff, maximum retention.",
    "longer": "Make the script more detailed, add deeper storytelling, stronger arguments, and 4-5 rich scenes.",
    "bolder": "Make the hook and voiceover much bolder, provocative, controversial, and scroll-stopping.",
    "native": "Make the voiceover sound 100% like a casual voice message to a best friend, zero ad cliches.",
    "funny": "Inject witty relatable humor, irony, and meme energy into the hook and voiceover.",
}
LANG_NAMES = {"ru": "Russian", "en": "English"}


def _parse_model_json(text, require_key: str = "scripts") -> dict:
    clean = (text or "").strip()
    if not clean:
        raise AppError("empty")

    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            raise AppError("bad_json")
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            raise AppError("bad_json")

    if isinstance(parsed, list):
        parsed = next((x for x in parsed if isinstance(x, dict)), None)
    if not isinstance(parsed, dict) or not parsed.get(require_key):
        raise AppError("bad_json")
    return parsed


async def _call_gemini_json(system_inst: str, user_prompt: str, require_key: str) -> dict:
    if not ai_client:
        raise AppError("no_key")

    config = genai_types.GenerateContentConfig(
        system_instruction=system_inst,
        temperature=0.8,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json"
    )

    models_to_try = await get_model_candidates()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TOTAL_BUDGET_SEC
    busy_error = None
    gone_error = None
    bad_answer = None
    parsed = None

    for model_name in models_to_try:
        if parsed is not None:
            break
        for attempt in range(3):
            if loop.time() > deadline:
                break
            try:
                resp = await asyncio.to_thread(
                    ai_client.models.generate_content,
                    model=model_name,
                    contents=user_prompt,
                    config=config
                )
                parsed = _parse_model_json(getattr(resp, "text", None), require_key=require_key)
                logging.info(f"Gemini OK: model={model_name}, attempt={attempt + 1}")
                break
            except AppError as ae:
                bad_answer = ae
                logging.warning(f"{model_name} bad answer ({ae.code}), attempt {attempt + 1}")
                continue
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in GONE_MARKERS):
                    gone_error = e
                    logging.warning(f"Model {model_name} unavailable, trying next. {msg[:160]}")
                    break
                if any(k in msg for k in TRANSIENT_MARKERS):
                    busy_error = e
                    logging.warning(f"{model_name} transient error (attempt {attempt + 1}): {msg[:160]}")
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise

    if parsed is None:
        if gone_error is not None:
            _MODEL_CACHE["ts"] = 0.0
        if busy_error is not None:
            raise busy_error
        if bad_answer is not None:
            raise bad_answer
        if gone_error is not None:
            raise gone_error
        raise AppError("busy")

    return parsed


async def generate_ugc_payload(data: dict, format_type: str, tone: str, lang: str = "ru") -> dict:
    lang = lang if lang in LANG_NAMES else "ru"
    user_prompt = f"""
PRODUCT DATA:
{json.dumps(data, ensure_ascii=False)}

VIDEO FORMAT: {FORMAT_LABELS.get(format_type, format_type)}
TONE: {TONE_LABELS.get(tone, tone)}
OUTPUT LANGUAGE: write ALL text fields in {LANG_NAMES[lang]} (except ai_video_prompt and video_generator_prompts, which are always in English).

REQUIREMENTS:
- 2 scripts with 4 scenes each (make voiceover sound natural for reading from a teleprompter).
- 5 viral hooks.
- 3 pain points, 3 killer features.
- 3 ready-to-publish Telegram ad posts (tg_posts) in different angles: native personal recommendation, aesthetic product find, and hot discount/offer.
- 2 video generator prompts.
"""
    parsed = await _call_gemini_json(SYSTEM_INSTRUCTION, user_prompt, require_key="scripts")
    parsed["meta_source"] = data.get("source", "Web")
    parsed["meta_image"] = data.get("image", "")
    parsed["meta_price"] = data.get("price", "")
    parsed["meta_brand"] = data.get("brand", "")
    parsed["meta_id"] = data.get("id", "")
    return parsed


async def rewrite_single_script(req: RewriteScriptRequest) -> dict:
    lang = req.lang if req.lang in LANG_NAMES else "ru"
    instruction = req.custom_instruction.strip() if req.action == "custom" and req.custom_instruction.strip() else REWRITE_ACTIONS.get(req.action, REWRITE_ACTIONS["shorter"])
    user_prompt = f"""
PRODUCT: {req.product_title}
CURRENT SCRIPT JSON:
{json.dumps(req.script, ensure_ascii=False)}

REWRITE INSTRUCTION:
{instruction}

OUTPUT LANGUAGE: {LANG_NAMES[lang]} (keep ai_video_prompt in English).
Return the updated script object inside {{"script": ...}}.
"""
    parsed = await _call_gemini_json(REWRITE_SYSTEM_INSTRUCTION, user_prompt, require_key="script")
    return parsed["script"]


async def generate_tg_posts_only(data: dict, style: str, lang: str = "ru") -> dict:
    lang = lang if lang in LANG_NAMES else "ru"
    user_prompt = f"""
PRODUCT DATA:
{json.dumps(data, ensure_ascii=False)}

PREFERRED TONE/STYLE: {style}
OUTPUT LANGUAGE: {LANG_NAMES[lang]}

Generate 3 distinct Telegram posts:
1) Native personal story/recommendation ("Девочки, нашла..." / "Честный тест...")
2) Short structured "Находка дня" with bullet points
3) Trigger/Offer post focusing on value and problem solving
"""
    parsed = await _call_gemini_json(TG_POST_SYSTEM_INSTRUCTION, user_prompt, require_key="tg_posts")
    parsed["meta_source"] = data.get("source", "Web")
    parsed["meta_image"] = data.get("image", "")
    parsed["meta_price"] = data.get("price", "")
    parsed["meta_brand"] = data.get("brand", "")
    parsed["meta_id"] = data.get("id", "")
    return parsed


_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_MAX = 64


async def run_pipeline(query: str, format_type: str, tone: str, lang: str) -> dict:
    key = ("ugc", query.strip().lower(), format_type, tone, lang)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    data = await extract_product_data(query)
    result = await generate_ugc_payload(data, format_type, tone, lang)

    _CACHE[key] = result
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return result


# =====================================================================
# TELEGRAM BOT (PREMIUM WELCOME + TELEGRAM STARS PAYMENTS)
# =====================================================================
BOT_TXT = {
    "ru": {
        "start": (
            "🎯 <b>SNIPER // UGC AI STUDIO 5.0</b>\n"
            "<i>Премиальная платформа создания вирального контента для TikTok, Reels, Shorts и Telegram</i>\n\n"
            "━━━━━━━━━━━━━━━\n"
            "⚡ <b>ЧТО УМЕЕТ СТУДИЯ:</b>\n"
            "• <b>Авто-парсинг в 1 клик:</b> Wildberries, Ozon и любые сайты — фото, бренд и цена подтягиваются мгновенно.\n"
            "• <b>Виральные сценарии + Рерайт:</b> хуки, покадровый план, AI-промпты и умные кнопки <i>«Короче / Дерзко / Нативнее»</i>.\n"
            "• <b>Встроенный Телесуфлёр:</b> читай текст прямо с экрана во время записи ролика.\n"
            "• <b>TG-Посты + ОРД Маркировка:</b> готовые рекламные посты с токеном <code>erid</code> и водяным знаком на фото товара.\n"
            "━━━━━━━━━━━━━━━\n"
            "💎 <b>ТАРИФЫ:</b>\n"
            "🎬 Полный разбор и сценарии — <b>100 ★</b>\n"
            "📢 Пакет рекламных TG-постов — <b>75 ★</b>\n"
            "🎁 Бонус новым авторам — <b>100 ★ на старте!</b>\n\n"
            "👇 <b>Отправь ссылку/артикул прямо в чат или открой студию:</b>"
        ),
        "open": "🚀 Запустить SNIPER STUDIO",
        "buy_100": "⭐ Купить 100 ★ (1 сценарий)",
        "buy_300": "🔥 Купить 300 ★ (Хит)",
        "how_btn": "💡 Как снимать с суфлёром",
        "how_text": (
            "🎬 <b>КАК СНИМАТЬ РОЛИКИ ЗА 2 МИНУТЫ:</b>\n\n"
            "1️⃣ Вставь ссылку на товар с Wildberries / Ozon в студию.\n"
            "2️⃣ Нажми <b>«Собрать сценарии»</b> — ИИ распишет хуки, кадры и озвучку.\n"
            "3️⃣ Нужно динамичнее? Нажми кнопку <b>«✂️ Короче»</b> или <b>«🔥 Дерзко»</b> прямо под сценарием.\n"
            "4️⃣ Нажми <b>«🎬 СУФЛЁР»</b>, поставь телефон перед собой и запиши видео с первого дубля!"
        ),
        "wait": "🔍 <b>Сканирую карточку товара и собираю виральную воронку...</b>",
        "hooks": "🔥 <b>ТОП-ХУКИ:</b>",
        "script": "🎬 <b>СЦЕНАРИЙ:",
        "viral": "виральность",
        "hook": "Хук:",
        "full": "⚡ Открыть рерайт и телесуфлёр",
        "err": "⚠️ Ошибка:",
        "paid": "✅ <b>Оплата прошла успешно!</b>\nЗвёзды зачислены на твой баланс в приложении SNIPER UGC.",
    },
    "en": {
        "start": (
            "🎯 <b>SNIPER // UGC AI STUDIO 5.0</b>\n"
            "<i>Premium Viral Content & Ad Production Suite for TikTok, Reels, Shorts & Telegram</i>\n\n"
            "━━━━━━━━━━━━━━━\n"
            "⚡ <b>STUDIO FEATURES:</b>\n"
            "• <b>1-Tap Product Parser:</b> Wildberries, Ozon & any web link with live product preview.\n"
            "• <b>Viral Scripts + AI Rewrite:</b> scroll-stopping hooks, storyboards & instant 1-tap script restyling.\n"
            "• <b>Fullscreen Teleprompter:</b> record videos in one take with eye-line guide & speed control.\n"
            "• <b>Telegram Ad Builder:</b> native ad posts with ERID disclosure & photo watermarks.\n"
            "━━━━━━━━━━━━━━━\n"
            "💎 <b>PRICING:</b>\n"
            "🎬 Full Script Funnel — <b>100 ★</b>\n"
            "📢 Telegram Ad Post Pack — <b>75 ★</b>\n"
            "🎁 Welcome Bonus — <b>100 ★ Free!</b>\n\n"
            "👇 <b>Send a link/SKU right here or launch the studio:</b>"
        ),
        "open": "🚀 Launch SNIPER STUDIO",
        "buy_100": "⭐ Top Up 100 ★ (1 Script)",
        "buy_300": "🔥 Top Up 300 ★ (Popular)",
        "how_btn": "💡 How it works",
        "how_text": (
            "🎬 <b>HOW TO SHOOT VIRAL VIDEOS IN 2 MINUTES:</b>\n\n"
            "1️⃣ Paste any product link or SKU into the studio.\n"
            "2️⃣ Generate hooks, storyboards, and voiceovers.\n"
            "3️⃣ Use 1-tap AI Rewrite buttons (<i>Shorter / Bolder / Native</i>) to tune the script.\n"
            "4️⃣ Tap <b>🎬 PROMPTER</b> and record your video in a single take!"
        ),
        "wait": "🔍 <b>Scanning product and building viral funnel...</b>",
        "hooks": "🔥 <b>TOP HOOKS:</b>",
        "script": "🎬 <b>SCRIPT:",
        "viral": "virality",
        "hook": "Hook:",
        "full": "⚡ Open AI Rewrite & Teleprompter",
        "err": "⚠️ Error:",
        "paid": "✅ <b>Payment successful!</b>\nStars have been added to your SNIPER UGC balance.",
    },
}


def user_lang(user_obj) -> str:
    code = getattr(user_obj, "language_code", "") if user_obj else ""
    return norm_lang(code)


@dp.pre_checkout_query()
async def on_pre_checkout(pre_checkout_query: PreCheckoutQuery):
    await pre_checkout_query.answer(ok=True)


@dp.message(F.successful_payment)
async def on_successful_payment(message: types.Message):
    L = BOT_TXT[user_lang(message.from_user)]
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=L["open"], web_app=WebAppInfo(url=APP_URL_VERSIONED))]
    ])
    await message.answer(L["paid"], reply_markup=kb, parse_mode="HTML")


@dp.message(Command("start"))
async def bot_start(message: types.Message):
    lang = user_lang(message.from_user)
    L = BOT_TXT[lang]
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=L["open"], web_app=WebAppInfo(url=APP_URL_VERSIONED))],
        [
            InlineKeyboardButton(text=L["buy_100"], callback_data="buy:pack_100"),
            InlineKeyboardButton(text=L["buy_300"], callback_data="buy:pack_300"),
        ],
        [InlineKeyboardButton(text=L["how_btn"], callback_data="how_it_works")],
    ])
    await message.answer(L["start"], reply_markup=kb, parse_mode="HTML")


@dp.callback_query(F.data == "how_it_works")
async def cb_how_it_works(call: CallbackQuery):
    lang = user_lang(call.from_user)
    L = BOT_TXT[lang]
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=L["open"], web_app=WebAppInfo(url=APP_URL_VERSIONED))]
    ])
    await call.message.answer(L["how_text"], reply_markup=kb, parse_mode="HTML")
    await call.answer()


@dp.callback_query(F.data.startswith("buy:"))
async def cb_buy_pack(call: CallbackQuery):
    pack_id = call.data.split(":", 1)[1]
    pack = STAR_PACKS.get(pack_id)
    if not pack:
        await call.answer("Пакет не найден", show_alert=True)
        return
    lang = user_lang(call.from_user)
    uid = call.from_user.id if call.from_user else 0
    await call.message.answer_invoice(
        title=pack[f"title_{lang}"],
        description=pack[f"desc_{lang}"],
        payload=f"{pack_id}:{uid}:{pack['credits']}",
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label=pack[f"title_{lang}"], amount=pack["stars"])],
    )
    await call.answer()


@dp.message()
async def bot_quick_message(message: types.Message):
    txt = (message.text or "").strip()
    if not txt:
        return
    lang = user_lang(message.from_user)
    L = BOT_TXT[lang]
    wait_msg = await message.answer(L["wait"], parse_mode="HTML")
    try:
        res = await run_pipeline(txt, "ugc", "native", lang)
        scripts = res.get("scripts") or []
        if not scripts:
            raise AppError("bad_json")
        s1 = scripts[0]

        hooks = "\n".join(
            f"• <i>«{h(str(x))}»</i>" for x in (res.get("viral_hooks") or [])[:3]
        )

        reply = (
            f"🎯 <b>{h(str(res.get('product_title', '')))}</b>\n\n"
            f"{L['hooks']}\n{hooks}\n\n"
            f"{L['script']} {h(str(s1.get('title', '')))} ({h(str(s1.get('virality_score', 95)))}% {L['viral']})</b>\n"
            f"<b>{L['hook']}</b> {h(str(s1.get('hook', '')))}\n\n"
            f"📢 <b>CTA:</b> {h(str(s1.get('cta', '')))}"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=L["full"], web_app=WebAppInfo(url=APP_URL_VERSIONED))]
        ])
        await wait_msg.edit_text(reply, reply_markup=kb, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Telegram handler error: {e}", exc_info=True)
        try:
            await wait_msg.edit_text(f"{L['err']} {h(error_text(e, lang))[:3500]}", parse_mode="HTML")
        except Exception:
            pass


# =====================================================================
# FASTAPI ENDPOINTS
# =====================================================================
async def keep_alive_loop():
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                await c.get(APP_URL.rstrip("/") + "/health")
        except Exception as e:
            logging.warning(f"keep-alive ping failed: {e}")
        await asyncio.sleep(600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    ka_task = asyncio.create_task(keep_alive_loop()) if KEEP_ALIVE else None
    if bot:
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            task = asyncio.create_task(dp.start_polling(bot, handle_signals=False))
            logging.info("Bot polling started.")
        except Exception as e:
            logging.error(f"Polling start failed: {e}")
    yield
    if ka_task:
        ka_task.cancel()
        await asyncio.gather(ka_task, return_exceptions=True)
    if task:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    if bot:
        await bot.session.close()


app = FastAPI(title="SNIPER UGC", lifespan=lifespan)


@app.head("/")
async def head_ping():
    return HTMLResponse(content="", status_code=200)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/api/preview")
async def api_preview_endpoint(req: PreviewRequest):
    q = (req.query or "").strip()
    if len(q) < 4:
        raise HTTPException(status_code=400, detail="Too short")
    data = await extract_product_data(q)
    return {
        "source": data.get("source", "Web"),
        "id": data.get("id", ""),
        "title": data.get("title", ""),
        "brand": data.get("brand", ""),
        "category": data.get("category", ""),
        "price": data.get("price", ""),
        "image": data.get("image", ""),
        "features": (data.get("features") or [])[:4],
    }


@app.get("/api/proxy-image")
async def api_proxy_image(url: str):
    if not url or not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid image URL")
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=8.0) as client:
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(status_code=404, detail="Image not found")
            content_type = r.headers.get("content-type", "image/jpeg")
            return Response(
                content=r.content,
                media_type=content_type,
                headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600"}
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def api_analyze_endpoint(req: AnalyzeRequest):
    lang = req.lang if req.lang in ERR else "en"
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail=ERR[lang]["short_query"])
    try:
        return await run_pipeline(req.query, req.format_type, req.tone, lang)
    except Exception as e:
        logging.error(f"API error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, lang))


@app.post("/api/rewrite-script")
async def api_rewrite_script_endpoint(req: RewriteScriptRequest):
    lang = req.lang if req.lang in ERR else "en"
    try:
        updated = await rewrite_single_script(req)
        return {"script": updated}
    except Exception as e:
        logging.error(f"Rewrite API error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, lang))


@app.post("/api/tg-post")
async def api_tg_post_endpoint(req: TgPostRequest):
    lang = req.lang if req.lang in ERR else "en"
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail=ERR[lang]["short_query"])
    try:
        data = await extract_product_data(req.query)
        return await generate_tg_posts_only(data, req.style, lang)
    except Exception as e:
        logging.error(f"TG Post API error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, lang))


@app.post("/api/create-stars-invoice")
async def api_create_stars_invoice(req: StarsInvoiceRequest):
    pack = STAR_PACKS.get(req.pack_id)
    if not pack:
        raise HTTPException(status_code=400, detail="Unknown package")
    if not bot:
        raise HTTPException(status_code=503, detail="BOT_TOKEN is not configured on server")
    lang = "ru" if req.lang == "ru" else "en"
    try:
        invoice_link = await bot.create_invoice_link(
            title=pack[f"title_{lang}"],
            description=pack[f"desc_{lang}"],
            payload=f"{req.pack_id}:{req.user_id}:{pack['credits']}",
            provider_token="",
            currency="XTR",
            prices=[LabeledPrice(label=pack[f"title_{lang}"], amount=pack["stars"])],
        )
        return {"invoice_link": invoice_link, "credits": pack["credits"], "stars": pack["stars"]}
    except Exception as e:
        logging.error(f"Stars invoice creation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stars invoice error: {e}")


# =====================================================================
# FRONTEND UI (ULTRA-PREMIUM DESIGN + AI REWRITE + TELEPROMPTER)
# =====================================================================
FRONTEND_UI = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<title>SNIPER // UGC STUDIO 5.0</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Unbounded:wght@500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #050508;
    --surface: rgba(16, 16, 23, 0.78);
    --surface-solid: #101017;
    --surface2: #171722;
    --surface3: #20202f;
    --border: rgba(255, 255, 255, 0.08);
    --border-highlight: rgba(255, 255, 255, 0.15);
    --border-strong: rgba(212, 255, 58, 0.48);
    --text: #f7f7fa;
    --muted: #8e8ea0;
    --accent: #d4ff3a;
    --accent2: #34d399;
    --accent-glow: rgba(212, 255, 58, 0.25);
    --accent-ink: #060608;
    --purple: #8b5cf6;
    --danger: #ff5c5c;
    --ok: #34d399;
    --radius: 20px;
    --font: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    --display: 'Unbounded', 'Manrope', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; line-height: 1.48; overflow-x: hidden; }
  body {
    min-height: 100vh;
    background-color: var(--bg);
    position: relative;
  }

  /* AMBIENT FLOATING AURORA BACKGROUND */
  body::before, body::after {
    content: ""; position: fixed; pointer-events: none; z-index: 0; border-radius: 50%; filter: blur(95px); opacity: 0.45;
    animation: auroraFloat 14s ease-in-out infinite alternate;
  }
  body::before {
    width: 320px; height: 320px; top: -90px; right: -70px;
    background: radial-gradient(circle, rgba(212, 255, 58, 0.28), transparent 70%);
  }
  body::after {
    width: 300px; height: 300px; top: 28%; left: -110px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.24), transparent 70%);
    animation-delay: -7s;
  }
  @keyframes auroraFloat {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(18px, 26px) scale(1.12); }
  }

  .app { position: relative; z-index: 1; max-width: 520px; margin: 0 auto; padding: 14px 16px calc(112px + env(safe-area-inset-bottom)); }
  .mono { font-family: var(--mono); }
  button { font-family: inherit; user-select: none; }

  /* TOPBAR */
  .topbar {
    display: flex; justify-content: space-between; align-items: center; gap: 8px;
    margin-bottom: 18px; padding: 6px 0;
  }
  .brand { display: flex; align-items: center; gap: 8px; font-family: var(--display); font-weight: 800; font-size: 14px; letter-spacing: 0.4px; cursor: pointer; }
  .brand .dot {
    width: 10px; height: 10px; border-radius: 50%; background: var(--accent);
    box-shadow: 0 0 16px var(--accent), 0 0 6px #fff; animation: pulseDot 2.2s infinite;
  }
  @keyframes pulseDot { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.28); opacity: 0.75; } }
  .brand .sub {
    color: var(--muted); font-weight: 600; font-size: 10px;
    background: rgba(255,255,255,0.04); padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border);
  }

  .dev-top-badge {
    display: none; align-items: center; gap: 4px;
    font-family: var(--mono); font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;
    color: #000; background: linear-gradient(135deg, #d4ff3a, #34d399);
    padding: 3px 8px; border-radius: 999px;
    box-shadow: 0 0 18px rgba(212, 255, 58, 0.5); text-transform: uppercase;
  }
  .dev-top-badge.active { display: inline-flex; }

  .top-actions { display: flex; align-items: center; gap: 8px; }
  .bal-chip {
    display: flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(212,255,58,0.16), rgba(255,255,255,0.04));
    border: 1px solid rgba(212,255,58,0.38); color: var(--text);
    padding: 7px 13px; border-radius: 999px; font-weight: 800; font-size: 13px; cursor: pointer;
    box-shadow: 0 4px 15px rgba(212, 255, 58, 0.12);
    transition: transform 0.15s cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .bal-chip:active { transform: scale(0.93); }
  .bal-chip .coin { color: var(--accent); font-size: 14px; }
  .lang-sw { display: flex; background: var(--surface-solid); border: 1px solid var(--border); border-radius: 999px; padding: 3px; }
  .lang-sw button {
    background: transparent; border: none; color: var(--muted); font-weight: 800; font-size: 11px;
    padding: 5px 9px; border-radius: 999px; cursor: pointer; transition: all 0.2s ease;
  }
  .lang-sw button.active { background: var(--text); color: #000; }

  /* HERO HEADER BANNER */
  .hero-box {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(212,255,58,0.11) 0%, rgba(139,92,246,0.09) 100%), var(--surface);
    border: 1px solid var(--border-highlight); border-radius: var(--radius);
    padding: 16px; margin-bottom: 14px;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  }
  .hero-kicker {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--mono); font-size: 10px; font-weight: 700; color: var(--accent);
    background: rgba(212,255,58,0.1); border: 1px solid rgba(212,255,58,0.28);
    padding: 3px 9px; border-radius: 999px; margin-bottom: 8px; text-transform: uppercase;
  }
  h1.title { font-family: var(--display); font-size: 18px; font-weight: 700; line-height: 1.25; margin-bottom: 5px; letter-spacing: -0.02em; }
  p.lead { color: var(--muted); font-size: 12.5px; line-height: 1.45; }

  /* VIEWS & STAGGER ANIMATIONS */
  .view { display: none; }
  .view.active { display: block; animation: viewEnter 0.35s cubic-bezier(0.16, 1, 0.3, 1); }
  @keyframes viewEnter { from { opacity: 0; transform: translateY(12px) scale(0.99); } to { opacity: 1; transform: none; } }

  .panel, .card {
    background: var(--surface); backdrop-filter: blur(22px); -webkit-backdrop-filter: blur(22px);
    border: 1px solid var(--border); border-top-color: var(--border-highlight);
    border-radius: var(--radius); padding: 16px; margin-bottom: 14px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.38);
    transition: border-color 0.25s ease, transform 0.2s ease;
  }
  .stagger-card { animation: cardPop 0.4s cubic-bezier(0.16, 1, 0.3, 1) both; }
  @keyframes cardPop { from { opacity: 0; transform: translateY(16px) scale(0.98); } to { opacity: 1; transform: none; } }

  .lbl { font-family: var(--mono); font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }

  /* SMART INPUT CAPSULE */
  .smart-input-pod {
    background: rgba(6, 6, 10, 0.9);
    border: 1.5px solid var(--border-highlight);
    border-radius: 16px; padding: 10px 12px; margin-bottom: 12px;
    transition: all 0.22s ease;
  }
  .smart-input-pod:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px rgba(212, 255, 58, 0.14);
  }
  .smart-input-row { display: flex; align-items: center; gap: 9px; }
  .smart-icon {
    width: 36px; height: 36px; border-radius: 11px;
    background: linear-gradient(135deg, rgba(212,255,58,0.15), rgba(255,255,255,0.03));
    border: 1px solid rgba(212,255,58,0.3);
    display: flex; align-items: center; justify-content: center;
    color: var(--accent); flex-shrink: 0; font-size: 16px;
  }
  .smart-field {
    flex: 1; min-width: 0; background: transparent; border: none; outline: none;
    color: var(--text); font-size: 14px; font-family: inherit; font-weight: 500;
    line-height: 1.38; resize: none; max-height: 90px; overflow-y: auto;
  }
  .smart-field::placeholder { color: #565668; font-weight: 400; }
  .smart-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .paste-pill {
    background: var(--surface2); color: var(--text); border: 1px solid var(--border-highlight);
    padding: 7px 11px; border-radius: 10px; font-size: 11px; font-weight: 800;
    cursor: pointer; display: flex; align-items: center; gap: 4px; transition: all 0.15s ease;
  }
  .paste-pill:active { transform: scale(0.93); background: var(--accent); color: #000; }
  .clear-pill {
    width: 28px; height: 28px; border-radius: 9px; background: var(--surface2);
    color: var(--muted); border: 1px solid var(--border); display: none;
    align-items: center; justify-content: center; font-size: 12px; cursor: pointer;
  }

  /* LIVE PRODUCT OVERVIEW CARD */
  .live-preview-wrap {
    display: none; margin-bottom: 14px;
    background: linear-gradient(135deg, rgba(212, 255, 58, 0.1), rgba(52, 211, 153, 0.06)), var(--surface2);
    border: 1px solid var(--border-strong); border-radius: 15px; padding: 11px 13px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    animation: slideDown 0.28s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes slideDown { from { opacity: 0; transform: translateY(-10px) scale(0.97); } to { opacity: 1; transform: none; } }
  .lp-inner { display: flex; align-items: center; gap: 12px; }
  .lp-thumb {
    width: 56px; height: 56px; border-radius: 12px; object-fit: cover;
    background: #0b0b0f; border: 1px solid var(--border-highlight); flex-shrink: 0;
  }
  .lp-body { flex: 1; min-width: 0; }
  .lp-badges { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap; }
  .lp-badge {
    font-family: var(--mono); font-size: 9.5px; font-weight: 800; text-transform: uppercase;
    padding: 2px 7px; border-radius: 6px; background: var(--accent); color: #000;
  }
  .lp-price {
    font-family: var(--mono); font-size: 10.5px; font-weight: 800;
    padding: 2px 7px; border-radius: 6px; background: rgba(52, 211, 153, 0.18); color: var(--ok);
  }
  .lp-title { font-weight: 800; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
  .lp-sub { font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }

  /* CHIPS & SHIMMER CTA BUTTONS */
  .chips { display: flex; gap: 6px; overflow-x: auto; margin-bottom: 14px; scrollbar-width: none; padding-bottom: 2px; }
  .chips::-webkit-scrollbar { display: none; }
  .chip {
    flex-shrink: 0; background: rgba(8, 8, 12, 0.85); color: var(--muted); border: 1px solid var(--border);
    border-radius: 999px; padding: 8px 14px; font-size: 12px; font-weight: 700; cursor: pointer;
    transition: all 0.2s cubic-bezier(0.2, 0.8, 0.2, 1);
  }
  .chip:active { transform: scale(0.94); }
  .chip.active {
    background: var(--accent); color: var(--accent-ink); border-color: var(--accent);
    box-shadow: 0 4px 16px var(--accent-glow);
  }

  .btn {
    position: relative; overflow: hidden;
    width: 100%; background: linear-gradient(135deg, #d4ff3a 0%, #b8f528 100%);
    color: var(--accent-ink); border: none; padding: 15px 18px;
    font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px;
    border-radius: 15px; cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 8px;
    box-shadow: 0 8px 25px var(--accent-glow); transition: transform 0.14s ease, opacity 0.15s ease;
  }
  .btn::after {
    content: ""; position: absolute; top: 0; left: -120%; width: 65%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.45), transparent);
    transform: skewX(-22deg); animation: btnShimmer 4.5s infinite;
  }
  @keyframes btnShimmer { 0%, 65% { left: -120%; } 100% { left: 160%; } }
  .btn:active { transform: scale(0.975); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
  .btn.ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border-highlight); box-shadow: none; }
  .btn.ghost::after { display: none; }
  .btn.small { width: auto; padding: 9px 14px; font-size: 11.5px; border-radius: 11px; box-shadow: none; }
  .cost-note { text-align: center; color: var(--muted); font-size: 11px; margin-top: 10px; font-family: var(--mono); }

  /* AI REWRITE TOOLBAR INSIDE EACH SCRIPT CARD */
  .rewrite-bar {
    margin-top: 12px; padding-top: 12px; border-top: 1px dashed var(--border-highlight);
  }
  .rewrite-lbl {
    font-family: var(--mono); font-size: 10px; color: var(--accent); text-transform: uppercase;
    letter-spacing: 0.6px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;
  }
  .rewrite-chips {
    display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px;
  }
  .rw-chip {
    background: var(--surface2); color: var(--text); border: 1px solid var(--border-highlight);
    padding: 6px 11px; border-radius: 999px; font-size: 11px; font-weight: 700; cursor: pointer;
    display: inline-flex; align-items: center; gap: 4px; transition: all 0.16s ease;
  }
  .rw-chip:hover, .rw-chip:active {
    background: var(--accent); color: #000; border-color: var(--accent); transform: translateY(-1px);
  }
  .rw-custom-box {
    display: none; gap: 6px; margin-top: 8px; animation: slideDown 0.2s ease;
  }
  .rw-custom-box.open { display: flex; }

  /* AI LIVE TERMINAL LOADER */
  .ai-terminal {
    display: none; background: #08080c; border: 1px solid var(--border-strong);
    border-radius: var(--radius); padding: 16px; margin-bottom: 16px;
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.55), 0 0 28px rgba(212, 255, 58, 0.1);
  }
  .term-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .term-dots { display: flex; gap: 6px; align-items: center; }
  .term-dots span { width: 8px; height: 8px; border-radius: 50%; }
  .term-dots span:nth-child(1) { background: #ff5f56; }
  .term-dots span:nth-child(2) { background: #ffbd2e; }
  .term-dots span:nth-child(3) { background: #27c93f; }
  .term-status { font-family: var(--mono); font-size: 11px; color: var(--accent); font-weight: 700; }
  .term-bar-bg { width: 100%; height: 6px; background: var(--surface2); border-radius: 999px; overflow: hidden; margin-bottom: 14px; }
  .term-bar-fill { height: 100%; width: 12%; background: linear-gradient(90deg, var(--accent), #34d399); border-radius: 999px; transition: width 0.4s ease; }
  .term-stream {
    font-family: var(--mono); font-size: 11.5px; line-height: 1.6; color: #d1d1e0;
    background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 12px;
    padding: 12px; min-height: 118px; max-height: 180px; overflow-y: auto;
  }
  .term-line { margin-bottom: 5px; display: flex; gap: 8px; align-items: flex-start; }
  .term-line .ok { color: var(--accent); flex-shrink: 0; }
  .term-cursor { display: inline-block; width: 7px; height: 13px; background: var(--accent); vertical-align: middle; margin-left: 4px; animation: blink 0.8s infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  /* RESULTS */
  #results { display: none; }
  .prod-head { display: flex; gap: 12px; align-items: center; padding: 14px; background: var(--surface); border: 1px solid var(--border-highlight); border-radius: var(--radius); margin-bottom: 14px; }
  .prod-thumb { width: 60px; height: 60px; border-radius: 14px; object-fit: cover; background: #111; flex-shrink: 0; border: 1px solid var(--border-highlight); }
  .tag { font-family: var(--mono); font-size: 10px; border: 1px solid var(--border-highlight); padding: 3px 8px; border-radius: 999px; color: var(--muted); background: var(--surface2); }

  .tabs {
    display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none;
    background: var(--surface-solid); padding: 5px; border: 1px solid var(--border-highlight); border-radius: 15px; margin-bottom: 14px;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tab-btn {
    flex: 1; white-space: nowrap; background: transparent; color: var(--muted); border: none;
    padding: 9px 10px; font-size: 10.5px; font-weight: 800; text-transform: uppercase; border-radius: 11px; cursor: pointer;
    transition: all 0.18s ease;
  }
  .tab-btn.active { background: var(--accent); color: #000; box-shadow: 0 3px 12px var(--accent-glow); }
  .tab-content { display: none; }
  .tab-content.active { display: block; animation: viewEnter 0.25s ease; }

  .card-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .action-row { display: flex; gap: 6px; align-items: center; }
  .copy-btn {
    background: var(--surface2); border: 1px solid var(--border-highlight); color: var(--text);
    padding: 6px 10px; border-radius: 9px; font-size: 10.5px; font-weight: 700; cursor: pointer;
    font-family: var(--mono); display: inline-flex; align-items: center; gap: 4px; transition: all 0.15s ease;
  }
  .copy-btn:active { background: var(--accent); color: #000; transform: scale(0.95); }
  .prompter-btn {
    background: linear-gradient(135deg, var(--accent), #a3e635); color: #000; border: none;
    padding: 6px 12px; border-radius: 9px; font-size: 10.5px; font-weight: 800; cursor: pointer;
    font-family: var(--mono); display: inline-flex; align-items: center; gap: 5px;
    box-shadow: 0 3px 12px var(--accent-glow); transition: transform 0.15s ease;
  }
  .prompter-btn:active { transform: scale(0.94); }

  .hook-box { background: rgba(5,5,8,0.85); border-left: 3px solid var(--accent); padding: 11px 13px; font-size: 13.5px; margin-bottom: 10px; border-radius: 0 12px 12px 0; }
  .scene-item { background: rgba(5,5,8,0.8); border: 1px solid var(--border); border-radius: 13px; padding: 11px 13px; margin-bottom: 8px; font-size: 12.5px; }
  .prompt-box { margin-top: 8px; padding: 8px 10px; background: var(--surface2); border: 1px dashed var(--border-highlight); border-radius: 10px; font-size: 11px; color: #b8b8c8; display: flex; justify-content: space-between; align-items: center; gap: 8px; }

  /* TG POST & ERID & WATERMARK STUDIO */
  .input-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
  input.inp {
    width: 100%; background: rgba(5,5,8,0.9); border: 1px solid var(--border-highlight); border-radius: 12px;
    color: var(--text); padding: 10px 12px; font-size: 13px; font-family: inherit; outline: none;
  }
  input.inp:focus { border-color: var(--accent); }
  .erid-preview {
    background: rgba(139, 92, 246, 0.1); border: 1px dashed rgba(139, 92, 246, 0.5);
    border-radius: 11px; padding: 9px 11px; font-size: 11.5px; color: #d8b4fe; margin-top: 8px;
    font-family: var(--mono);
  }
  .tg-post-body { white-space: pre-line; font-size: 13.5px; line-height: 1.52; margin: 8px 0 10px; color: #ececf3; }
  .wm-canvas-wrap {
    width: 100%; background: #050508; border: 1px solid var(--border-highlight); border-radius: 14px;
    overflow: hidden; margin: 10px 0; display: flex; align-items: center; justify-content: center; min-height: 220px;
  }
  canvas#wmCanvas { max-width: 100%; max-height: 340px; display: block; border-radius: 10px; }

  /* DEVELOPER CONSOLE CARD */
  .dev-console {
    display: none;
    background: linear-gradient(135deg, rgba(212, 255, 58, 0.14), rgba(52, 211, 153, 0.08)), #0b0b10;
    border: 1.5px solid var(--accent); border-radius: var(--radius);
    padding: 15px; margin-bottom: 14px; box-shadow: 0 0 28px rgba(212, 255, 58, 0.16);
  }
  .dev-console.active { display: block; }
  .dev-console-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid rgba(212, 255, 58, 0.2); }
  .dev-console-title { font-family: var(--mono); font-size: 11.5px; font-weight: 800; color: var(--accent); letter-spacing: 0.6px; display: flex; align-items: center; gap: 6px; }
  .dev-actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }

  /* FULLSCREEN TELEPROMPTER */
  .teleprompter-modal {
    position: fixed; inset: 0; z-index: 500; background: #000000;
    display: none; flex-direction: column; justify-content: space-between;
    padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom);
  }
  .teleprompter-modal.open { display: flex; animation: viewEnter 0.22s ease; }
  .tp-topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; background: rgba(15, 15, 21, 0.92); border-bottom: 1px solid var(--border); z-index: 5;
  }
  .tp-title { font-family: var(--mono); font-size: 12px; color: var(--accent); font-weight: 700; display: flex; align-items: center; gap: 6px; }
  .tp-stage { position: relative; flex: 1; overflow-y: auto; padding: 38vh 22px 45vh; scroll-behavior: auto; }
  .tp-stage.mirror { transform: scaleX(-1); }
  .tp-eye-guide {
    position: fixed; top: 36%; left: 0; right: 0; height: 92px; pointer-events: none; z-index: 3;
    border-top: 1.5px dashed rgba(212, 255, 58, 0.48);
    border-bottom: 1.5px dashed rgba(212, 255, 58, 0.48);
    background: linear-gradient(90deg, rgba(212,255,58,0.06), transparent 15%, transparent 85%, rgba(212,255,58,0.06));
  }
  .tp-eye-tag {
    position: fixed; top: calc(36% - 18px); left: 16px; z-index: 4; pointer-events: none;
    font-family: var(--mono); font-size: 9px; color: var(--accent); text-transform: uppercase; letter-spacing: 1px;
  }
  .tp-text { font-size: 28px; font-weight: 800; line-height: 1.45; color: #ffffff; text-align: center; outline: none; white-space: pre-wrap; }
  .tp-countdown {
    position: fixed; inset: 0; z-index: 10; background: rgba(0,0,0,0.85);
    display: none; align-items: center; justify-content: center;
    font-family: var(--display); font-size: 86px; font-weight: 800; color: var(--accent);
  }
  .tp-controls { background: rgba(15, 15, 21, 0.96); border-top: 1px solid var(--border); padding: 12px 16px 16px; z-index: 5; }
  .tp-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
  .tp-slider { flex: 1; accent-color: var(--accent); }

  /* CABINET & STARS */
  .profile { display: flex; align-items: center; gap: 12px; }
  .avatar { width: 54px; height: 54px; border-radius: 50%; background: var(--surface2); border: 1.5px solid var(--border-strong); display: flex; align-items: center; justify-content: center; font-family: var(--display); font-size: 18px; font-weight: 700; color: var(--accent); overflow: hidden; flex-shrink: 0; }
  .avatar img { width: 100%; height: 100%; object-fit: cover; }
  .balance-card {
    background: linear-gradient(135deg, rgba(212,255,58,0.16), rgba(139,92,246,0.14)), var(--surface);
    border: 1px solid var(--border-strong); border-radius: 22px; padding: 18px; margin-bottom: 14px;
  }
  .bal-num { font-family: var(--display); font-size: 36px; font-weight: 800; line-height: 1.1; margin: 6px 0 2px; color: var(--accent); }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 13px; }
  .stat .v { font-family: var(--display); font-size: 20px; font-weight: 700; }
  .stat .k { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .sec-title { font-family: var(--display); font-size: 13px; font-weight: 700; margin: 18px 0 10px; }
  .packs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
  .pack { background: var(--surface); border: 1px solid var(--border-highlight); border-radius: 16px; padding: 14px 8px; text-align: center; cursor: pointer; position: relative; color: var(--text); transition: transform 0.15s ease; }
  .pack:active { transform: scale(0.95); }
  .pack.pop { border-color: var(--accent); background: linear-gradient(180deg, rgba(212,255,58,0.12), var(--surface)); box-shadow: 0 6px 20px rgba(212,255,58,0.12); }
  .pack .pc { font-family: var(--display); font-size: 18px; font-weight: 800; }
  .pack .pk { color: var(--muted); font-size: 10px; margin-bottom: 8px; }
  .pack .pp { font-weight: 800; font-size: 12.5px; color: var(--accent); }
  .pack .badge { position: absolute; top: -8px; left: 50%; transform: translateX(-50%); background: var(--accent); color: #000; font-size: 9px; font-weight: 800; padding: 2px 8px; border-radius: 999px; text-transform: uppercase; }
  .list-item { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--border); }
  .list-item:last-child { border-bottom: none; }
  .li-main { min-width: 0; }
  .li-title { font-weight: 700; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .li-sub { color: var(--muted); font-size: 11px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .plus { color: var(--ok); font-weight: 800; }
  .minus { color: var(--danger); font-weight: 800; }
  .empty { color: var(--muted); font-size: 12px; text-align: center; padding: 14px 0; }
  .err-box { display: none; background: rgba(255,92,92,0.12); border: 1px solid rgba(255,92,92,0.4); color: #ffb3b3; border-radius: 14px; padding: 12px; margin-bottom: 14px; font-size: 13px; }

  /* FLOATING GLASS BOTTOM NAV */
  .nav {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 50;
    padding: 8px 14px calc(10px + env(safe-area-inset-bottom));
    background: linear-gradient(180deg, rgba(5,5,8,0) 0%, rgba(5,5,8,0.92) 35%);
    pointer-events: none;
  }
  .nav-inner {
    pointer-events: auto;
    max-width: 500px; margin: 0 auto; display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;
    background: rgba(18, 18, 26, 0.88); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-highlight); border-radius: 20px; padding: 6px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.55);
  }
  .nav button {
    background: transparent; border: none; color: var(--muted); display: flex; flex-direction: column; align-items: center; gap: 3px;
    font-size: 10.5px; font-weight: 800; padding: 7px 4px; border-radius: 14px; cursor: pointer; transition: all 0.2s ease;
  }
  .nav button.active { color: #000; background: var(--accent); box-shadow: 0 4px 14px var(--accent-glow); }
  .nav svg { width: 20px; height: 20px; stroke: currentColor; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }

  /* SHEET MODAL */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.72); backdrop-filter: blur(6px); z-index: 100; display: none; align-items: flex-end; justify-content: center; }
  .overlay.open { display: flex; }
  .sheet { width: 100%; max-width: 520px; background: var(--surface-solid); border: 1px solid var(--border-highlight); border-radius: 24px 24px 0 0; padding: 22px 18px calc(24px + env(safe-area-inset-bottom)); animation: up 0.24s cubic-bezier(0.16, 1, 0.3, 1); }
  @keyframes up { from { transform: translateY(40px); opacity: 0; } to { transform: none; opacity: 1; } }
  .sheet h3 { font-family: var(--display); font-size: 16px; font-weight: 700; margin-bottom: 12px; }
  .sheet .kv { display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
  .sheet p { color: var(--muted); font-size: 12px; margin: 10px 0 14px; }
  .sheet .btn + .btn { margin-top: 8px; }

  .toast {
    position: fixed; bottom: calc(96px + env(safe-area-inset-bottom)); left: 50%; transform: translateX(-50%);
    background: var(--accent); color: #000; padding: 10px 18px; border-radius: 999px; font-size: 12px; font-weight: 800;
    display: none; z-index: 600; white-space: nowrap; box-shadow: 0 8px 24px rgba(0,0,0,0.5);
  }
</style>
</head>
<body>
<div class="app">

  <!-- TOPBAR -->
  <div class="topbar">
    <div class="brand" onclick="onBrandSecretTap()">
      <span class="dot"></span>
      <span>SNIPER</span>
      <span id="normalStudioSub" class="sub">PRO 5.0</span>
      <span id="devTopBadge" class="dev-top-badge">⚡ DEVELOPER MODE</span>
    </div>
    <div class="top-actions">
      <button class="bal-chip" onclick="showView('cabinet')"><span class="coin">★</span><span id="balChip">0</span></button>
      <div class="lang-sw">
        <button data-l="ru" onclick="setLang('ru')">RU</button>
        <button data-l="en" onclick="setLang('en')">EN</button>
      </div>
    </div>
  </div>

  <!-- ==================== VIEW 1: STUDIO (VIDEO SCRIPTS + AI REWRITE) ==================== -->
  <section id="view-studio" class="view active">
    <div class="hero-box">
      <div class="hero-kicker"><span>✨</span><span>AI VIRAL PRODUCTION ENGINE</span></div>
      <h1 class="title" data-i18n="studio_title"></h1>
      <p class="lead" data-i18n="studio_lead"></p>
    </div>

    <div class="panel">
      <div class="lbl">
        <span data-i18n="src_label"></span>
        <span class="mono" style="color:var(--accent);">WB • OZON • WEB</span>
      </div>

      <!-- SMART INPUT POD -->
      <div class="smart-input-pod">
        <div class="smart-input-row">
          <div class="smart-icon">🔗</div>
          <textarea id="queryInp" class="smart-field" rows="2" data-i18n-ph="src_ph" oninput="onSmartInput(this, 'main')"></textarea>
          <div class="smart-actions">
            <button type="button" class="paste-pill" onclick="pasteClipboard('queryInp', 'main')">📋 <span data-i18n="paste"></span></button>
            <button type="button" id="clearMainBtn" class="clear-pill" onclick="clearSmartInput('queryInp', 'main')">✕</button>
          </div>
        </div>
      </div>

      <!-- AUTO LIVE PRODUCT PREVIEW -->
      <div id="livePreviewMain" class="live-preview-wrap"></div>

      <span class="lbl" data-i18n="fmt_label"></span>
      <div class="chips" id="fmtChips">
        <button class="chip active" data-v="ugc" data-i18n="f_ugc"></button>
        <button class="chip" data-v="problem_solution" data-i18n="f_ps"></button>
        <button class="chip" data-v="unboxing" data-i18n="f_unb"></button>
        <button class="chip" data-v="aesthetic" data-i18n="f_aes"></button>
        <button class="chip" data-v="storytelling" data-i18n="f_story"></button>
      </div>

      <span class="lbl" data-i18n="tone_label"></span>
      <div class="chips" id="toneChips">
        <button class="chip active" data-v="native" data-i18n="t_native"></button>
        <button class="chip" data-v="aggressive" data-i18n="t_aggr"></button>
        <button class="chip" data-v="humor" data-i18n="t_humor"></button>
        <button class="chip" data-v="luxury" data-i18n="t_lux"></button>
      </div>

      <button id="submitBtn" class="btn" onclick="startGeneration()">
        <span>⚡</span><span data-i18n="generate"></span><span id="scriptCostBtnBadge">(100 ★)</span>
      </button>
      <div class="cost-note" id="costNote"></div>
    </div>

    <div id="errBox" class="err-box"></div>

    <!-- AI LIVE STREAMING TERMINAL -->
    <div id="aiTerminal" class="ai-terminal">
      <div class="term-head">
        <div class="term-dots"><span></span><span></span><span></span></div>
        <div class="term-status" id="termPercent">AI ENGINE // 12%</div>
      </div>
      <div class="term-bar-bg"><div class="term-bar-fill" id="termBar"></div></div>
      <div class="term-stream" id="termStream"></div>
    </div>

    <!-- RESULTS -->
    <div id="results">
      <div class="prod-head stagger-card" id="pHeader"></div>
      <div class="tabs">
        <button class="tab-btn active" onclick="setTab('scripts', this)" data-i18n="tab_scripts"></button>
        <button class="tab-btn" onclick="setTab('tgposts', this)" data-i18n="tab_tgposts"></button>
        <button class="tab-btn" onclick="setTab('hooks', this)" data-i18n="tab_hooks"></button>
        <button class="tab-btn" onclick="setTab('prompts', this)" data-i18n="tab_video"></button>
        <button class="tab-btn" onclick="setTab('seo', this)" data-i18n="tab_seo"></button>
      </div>
      <div id="tab-scripts" class="tab-content active"></div>
      <div id="tab-tgposts" class="tab-content"></div>
      <div id="tab-hooks" class="tab-content"></div>
      <div id="tab-prompts" class="tab-content"></div>
      <div id="tab-seo" class="tab-content"></div>
    </div>
  </section>

  <!-- ==================== VIEW 2: TG AD POSTS + ERID + WATERMARK ==================== -->
  <section id="view-tgads" class="view">
    <div class="hero-box">
      <div class="hero-kicker"><span>📢</span><span>TELEGRAM AD SEEDING & ERID</span></div>
      <h1 class="title" data-i18n="tg_title"></h1>
      <p class="lead" data-i18n="tg_lead"></p>
    </div>

    <div class="panel">
      <span class="lbl" data-i18n="src_label"></span>
      <div class="smart-input-pod">
        <div class="smart-input-row">
          <div class="smart-icon">📢</div>
          <textarea id="tgQueryInp" class="smart-field" rows="2" data-i18n-ph="src_ph" oninput="onSmartInput(this, 'tg')"></textarea>
          <div class="smart-actions">
            <button type="button" class="paste-pill" onclick="pasteClipboard('tgQueryInp', 'tg')">📋 <span data-i18n="paste"></span></button>
            <button type="button" id="clearTgBtn" class="clear-pill" onclick="clearSmartInput('tgQueryInp', 'tg')">✕</button>
          </div>
        </div>
      </div>

      <div id="livePreviewTg" class="live-preview-wrap"></div>

      <!-- ERID & ORD MARKING BLOCK -->
      <div class="lbl" style="margin-top:4px;">
        <span data-i18n="erid_title"></span>
        <span class="mono" style="color:#d8b4fe; font-size:9.5px;">ФЗ-38 ОРД</span>
      </div>
      <div class="input-grid">
        <input id="eridOrgInp" class="inp" placeholder="ИП / ООО и ИНН (напр. ООО Ромашка)" oninput="updateEridPreview()">
        <input id="eridTokenInp" class="inp mono" placeholder="erid: 2Vtzqv..." oninput="updateEridPreview()">
      </div>
      <input id="eridLinkInp" class="inp" placeholder="Партнёрская ссылка или артикул WB/Ozon (необязательно)" oninput="updateEridPreview()" style="margin-bottom:10px;">
      <div id="eridLiveBox" class="erid-preview" style="display:none;"></div>

      <button id="tgGenBtn" class="btn" style="margin-top:12px;" onclick="generateStandaloneTgPosts()">
        <span>✨</span><span data-i18n="tg_generate"></span><span id="tgCostLabel">(75 ★)</span>
      </button>
    </div>

    <div id="tgPostsOutput"></div>

    <!-- WATERMARK STUDIO FOR PRODUCT PHOTOS -->
    <div class="panel">
      <div class="lbl">
        <span data-i18n="wm_title"></span>
        <span class="mono" style="color:var(--accent);">WATERMARK + ERID</span>
      </div>
      <p style="color:var(--muted); font-size:12px; margin-bottom:10px;" data-i18n="wm_desc"></p>

      <div class="input-grid">
        <input id="wmTextInp" class="inp" placeholder="@мой_канал или Водяной знак" value="@sniper_ugc" oninput="drawWatermark()">
        <input id="wmEridInp" class="inp mono" placeholder="Реклама • erid: ..." oninput="drawWatermark()">
      </div>

      <div class="chips" id="wmPosChips" style="margin-bottom:8px;">
        <button class="chip active" data-v="br" onclick="setWmPos('br', this)">Справа внизу</button>
        <button class="chip" data-v="bl" onclick="setWmPos('bl', this)">Слева внизу</button>
        <button class="chip" data-v="center" onclick="setWmPos('center', this)">По центру (диагональ)</button>
        <button class="chip" data-v="tl" onclick="setWmPos('tl', this)">Сверху</button>
      </div>

      <div class="wm-canvas-wrap">
        <canvas id="wmCanvas" width="600" height="750"></canvas>
      </div>

      <div style="display:flex; gap:8px;">
        <label class="btn ghost" style="flex:1; cursor:pointer;">
          <span>📷 Своё фото</span>
          <input type="file" accept="image/*" style="display:none;" onchange="uploadCustomWmImage(event)">
        </label>
        <button class="btn" style="flex:1;" onclick="downloadWatermarkedImage()">
          <span>⬇ Скачать фото</span>
        </button>
      </div>
    </div>
  </section>

  <!-- ==================== VIEW 3: CABINET & STARS ==================== -->
  <section id="view-cabinet" class="view">
    <div class="card"><div class="profile" id="cabProfile"></div></div>

    <!-- PERSONAL DEVELOPER CONSOLE -->
    <div id="devConsoleCard" class="dev-console">
      <div class="dev-console-head">
        <div class="dev-console-title"><span>⚡</span><span>DEVELOPER MODE // ROOT ACCESS</span></div>
        <span class="tag" style="background:var(--accent); color:#000; font-weight:800; border:none;">ACTIVE</span>
      </div>
      <div style="font-size:12px; color:#d1d1e0; line-height:1.45;">
        Персональный режим владельца активирован. Все генерации сценариев (100 ★), рерайты и TG-посты (75 ★) работают <strong>бесплатно и без ограничений (∞)</strong>.
      </div>
      <div class="dev-actions-grid">
        <button class="btn small" onclick="devAddCredits(1000)">+1000 ★ НА БАЛАНС</button>
        <button class="btn ghost small" onclick="toggleDevMode(false)">ОТКЛЮЧИТЬ DEV MODE</button>
      </div>
    </div>

    <div class="balance-card">
      <span class="lbl" data-i18n="balance"></span>
      <div class="bal-num" id="cabBalance">0</div>
      <div style="color:var(--muted); font-size:12px; margin-bottom:14px;" data-i18n="credits"></div>
      <button class="btn" onclick="scrollToTopup()"><span>⭐</span><span data-i18n="topup"></span></button>
    </div>

    <div class="stats" id="cabStats"></div>

    <div class="sec-title" id="topupSec" data-i18n="topup_title"></div>
    <div class="packs" id="cabPacks"></div>

    <div class="card">
      <span class="lbl" data-i18n="promo"></span>
      <div style="display:flex; gap:8px;">
        <input id="promoInp" class="inp" data-i18n-ph="promo_ph" autocapitalize="characters" autocomplete="off">
        <button class="btn small" onclick="applyPromo()" data-i18n="apply"></button>
      </div>
    </div>

    <div class="sec-title" style="display:flex; justify-content:space-between; align-items:center;">
      <span data-i18n="history"></span>
      <button style="background:none; border:none; color:var(--muted); font-size:11px; text-decoration:underline; cursor:pointer;" onclick="clearHistory()" data-i18n="clear"></button>
    </div>
    <div class="card" id="cabHistory"></div>

    <div class="sec-title" data-i18n="transactions"></div>
    <div class="card" id="cabTx"></div>
  </section>

</div>

<!-- ==================== FULLSCREEN TELEPROMPTER STUDIO ==================== -->
<div id="teleprompterModal" class="teleprompter-modal">
  <div class="tp-topbar">
    <div class="tp-title"><span>🔴</span><span>TELEPROMPTER STUDIO</span></div>
    <div style="display:flex; gap:6px;">
      <button class="copy-btn" onclick="toggleTpMirror()" id="tpMirrorBtn">⇄ MIRROR</button>
      <button class="copy-btn" onclick="resetTeleprompter()">↺</button>
      <button class="copy-btn" style="border-color:var(--danger); color:var(--danger);" onclick="closeTeleprompter()">✕ ЗАКРЫТЬ</button>
    </div>
  </div>

  <div class="tp-eye-tag">👀 ЗОНА ВЗГЛЯДА КАМЕРЫ</div>
  <div class="tp-eye-guide"></div>
  <div id="tpCountdown" class="tp-countdown">3</div>

  <div id="tpStage" class="tp-stage" onclick="toggleTeleprompterPlay()">
    <div id="tpText" class="tp-text" contenteditable="true" spellcheck="false"></div>
  </div>

  <div class="tp-controls">
    <div class="tp-row">
      <span class="mono" style="font-size:11px; color:var(--muted); width:95px;">СКОРОСТЬ: <strong style="color:#fff" id="tpSpeedVal">1.2x</strong></span>
      <input type="range" class="tp-slider" id="tpSpeedRange" min="0.4" max="3.0" step="0.1" value="1.2" oninput="onTpSpeedChange(this.value)">
      <button class="copy-btn" onclick="changeTpFont(-3)">A-</button>
      <button class="copy-btn" onclick="changeTpFont(3)">A+</button>
    </div>
    <button id="tpPlayBtn" class="btn" onclick="startTpWithCountdown()">▶ СТАРТ ЗАПИСИ (ОТЧЁТ 3 СЕК)</button>
  </div>
</div>

<!-- FLOATING GLASS BOTTOM NAVIGATION -->
<nav class="nav">
  <div class="nav-inner">
    <button id="nav-studio" class="active" onclick="showView('studio')">
      <svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z"/><path d="M19 17l.8 2.2L22 20l-2.2.8L19 23l-.8-2.2L16 20l2.2-.8L19 17z"/></svg>
      <span data-i18n="nav_studio"></span>
    </button>
    <button id="nav-tgads" onclick="showView('tgads')">
      <svg viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
      <span data-i18n="nav_tgads"></span>
    </button>
    <button id="nav-cabinet" onclick="showView('cabinet')">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></svg>
      <span data-i18n="nav_cabinet"></span>
    </button>
  </div>
</nav>

<div id="overlay" class="overlay" onclick="if(event.target===this)closeModal()">
  <div class="sheet" id="sheet"></div>
</div>

<div id="toast" class="toast"></div>

<script>
  // ================= Telegram Setup =================
  const SERVER_ADMIN_ID = "__ADMIN_TG_ID__";
  const tg = (window.Telegram && window.Telegram.WebApp) ? window.Telegram.WebApp : null;
  if (tg) {
    try {
      tg.ready();
      tg.expand();
      if (tg.setHeaderColor) tg.setHeaderColor('#050508');
      if (tg.setBackgroundColor) tg.setBackgroundColor('#050508');
    } catch (e) {}
  }
  const tgUser = (tg && tg.initDataUnsafe && tg.initDataUnsafe.user) ? tg.initDataUnsafe.user : null;

  function checkIsDefaultOwner() {
    if (!tgUser) return false;
    if (SERVER_ADMIN_ID && String(tgUser.id) === String(SERVER_ADMIN_ID)) return true;
    const fullName = [tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ').toLowerCase();
    const uname = (tgUser.username || '').toLowerCase();
    if (
      (fullName.includes('aleksandr') && fullName.includes('druzhinin')) ||
      (fullName.includes('александр') && fullName.includes('дружинин')) ||
      uname.includes('druzhinin') ||
      uname.includes('sniperugc')
    ) {
      return true;
    }
    return false;
  }

  // ================= Localization =================
  const I18N = {
    ru: {
      nav_studio: 'Сценарии', nav_tgads: 'TG Посты & ERID', nav_cabinet: 'Кабинет',
      studio_title: 'Виральные сценарии, AI-Рерайт и Суфлёр',
      studio_lead: 'Вставь ссылку WB / Ozon — получи сценарии с быстрым рерайтом («Короче / Дерзко / Нативнее») и встроенным суфлёром.',
      src_label: 'Товар (ссылка или артикул)',
      src_ph: 'Вставь ссылку WB, Ozon или артикул...',
      paste: 'Вставить',
      fmt_label: 'Формат ролика', tone_label: 'Тональность подачи',
      f_ugc: 'UGC обзор', f_ps: 'Боль → Решение', f_unb: 'Распаковка / ASMR', f_aes: 'Aesthetic', f_story: 'Сторителлинг',
      t_native: 'Нативный', t_aggr: 'Дерзкий', t_humor: 'Юмор', t_lux: 'Премиум',
      generate: 'Создать сценарии', cost: 'Стоимость генерации сценария: {n} ★ (AI-рерайты внутри бесплатно)',
      cost_dev: '⚡ DEVELOPER MODE: БЕСПЛАТНО (БЕЗЛИМИТ ∞)',
      tab_scripts: 'Сценарии', tab_tgposts: 'TG Посты', tab_hooks: 'Хуки', tab_video: 'AI Видео', tab_seo: 'ЦА & SEO',
      frame: 'Кадр', voice: 'Озвучка', screen_text: 'Текст', hook: 'ХУК', cta: 'CTA',
      copy: 'КОПИРОВАТЬ', copy_all: 'ВЕСЬ СЦЕНАРИЙ', copy_seo: 'КОПИРОВАТЬ SEO', copied: 'Скопировано в буфер!',
      prompter: '🎬 СУФЛЁР',
      rw_title: '✨ БЫСТРЫЙ AI-РЕРАЙТ СЦЕНАРИЯ (БЕСПЛАТНО):',
      rw_shorter: '✂️ Короче', rw_longer: '📝 Подробнее', rw_bolder: '🔥 Дерзко', rw_native: '💬 Нативнее', rw_funny: '😂 С юмором', rw_custom: '✍️ Своя правка',
      rw_ph: 'Напр.: сделай от лица девушки / добавь акцент на скидку 50%...',
      rw_go: 'Переписать', rw_loading: '⚡ AI переписывает сценарий...',
      audience: 'Целевая аудитория', pains: 'Боли ЦА', features: 'Главные фишки', seo_title: 'SEO описание и хэштеги',
      virality: 'виральность', script: 'Сценарий',
      tg_title: 'Рекламные TG-посты & Маркировка',
      tg_lead: 'Нативные посты для блогеров и каналов с авто-маркировкой (erid) и водяным знаком на фото.',
      erid_title: 'Маркировка рекламы (ОРД / ERID)',
      tg_generate: 'Сгенерировать 3 поста',
      wm_title: 'Водяной знак и ERID на фото товара',
      wm_desc: 'Автоматически берёт фото спарсенного товара (или загрузи своё) и накладывает твой ник и токен erid.',
      balance: 'Твой баланс', credits: '★ Stars на балансе (1 сценарий = 100 ★, 1 пост = 75 ★)', topup: 'Пополнить баланс через Telegram Stars',
      topup_title: 'Пакеты пополнения (Telegram Stars ★)',
      gens: 'Создано проектов', spent: 'Потрачено ★',
      pack_credits: '★ на баланс', popular: 'Хит выбора',
      promo: 'Промокод', promo_ph: 'Введи промокод (напр. SNIPER)', apply: 'Применить',
      promo_ok: 'Промокод активирован: +{n} ★', promo_bad: 'Неверный промокод', promo_used: 'Промокод уже использован',
      history: 'История проектов', clear: 'Очистить', history_empty: 'Пока пусто. Сгенерируй свой первый сценарий.',
      transactions: 'История операций', tx_empty: 'Операций пока нет',
      tx_bonus: 'Приветственный бонус', tx_topup: 'Пополнение Stars ★', tx_gen: 'Генерация сценария', tx_tg: 'Генерация TG-поста', tx_promo: 'Промокод',
      open: 'Открыть', plan: 'Creator PRO', guest: 'Креатор',
      pay_title: 'Пополнение баланса ★', pay_pack: 'Зачислится на баланс', pay_price: 'К оплате',
      pay_note: 'Официальная безопасная оплата через Telegram Stars (★). Звёзды зачисляются на баланс мгновенно.',
      pay_stars_btn: '⭐ Оплатить {s} Stars',
      pay_demo_btn: '⚡ Тестовое пополнение (Демо)',
      cancel: 'Отмена', processing: 'Создаём счёт...',
      pay_done: 'Баланс пополнен: +{n} ★!',
      nofunds_title: 'Недостаточно ★ на балансе',
      nofunds_text: 'Для этого действия нужно {n} ★ (1 сценарий = 100 ★, TG-пост = 75 ★). Пополни баланс за пару секунд.',
      err_query: 'Вставь ссылку, артикул или описание товара',
      err_timeout: 'Сервер отвечает слишком долго. Попробуй ещё раз.',
      err_network: 'Ошибка соединения. Попробуй ещё раз.',
      err_generic: 'Не удалось обработать запрос',
      cleared: 'История очищена'
    },
    en: {
      nav_studio: 'Scripts', nav_tgads: 'TG Ads & ERID', nav_cabinet: 'Account',
      studio_title: 'Viral Scripts, AI Rewrite & Teleprompter',
      studio_lead: 'Paste a WB / Ozon / web link — get scripts with 1-tap AI Rewrite ("Shorter / Bolder / Native") and Teleprompter.',
      src_label: 'Product (link or SKU)',
      src_ph: 'Paste WB, Ozon or website link / SKU...',
      paste: 'Paste',
      fmt_label: 'Video format', tone_label: 'Tone',
      f_ugc: 'UGC review', f_ps: 'Pain → Solution', f_unb: 'Unboxing / ASMR', f_aes: 'Aesthetic', f_story: 'Storytelling',
      t_native: 'Native', t_aggr: 'Bold', t_humor: 'Humor', t_lux: 'Premium',
      generate: 'Generate Scripts', cost: 'Script generation cost: {n} ★ (AI rewrites inside are free)',
      cost_dev: '⚡ DEVELOPER MODE: FREE (UNLIMITED ∞)',
      tab_scripts: 'Scripts', tab_tgposts: 'TG Posts', tab_hooks: 'Hooks', tab_video: 'AI Video', tab_seo: 'Audience & SEO',
      frame: 'Frame', voice: 'Voice', screen_text: 'Text', hook: 'HOOK', cta: 'CTA',
      copy: 'COPY', copy_all: 'FULL SCRIPT', copy_seo: 'COPY SEO', copied: 'Copied to clipboard!',
      prompter: '🎬 PROMPTER',
      rw_title: '✨ 1-TAP AI SCRIPT REWRITE (FREE):',
      rw_shorter: '✂️ Shorter', rw_longer: '📝 Detailed', rw_bolder: '🔥 Bolder', rw_native: '💬 More Native', rw_funny: '😂 Funny', rw_custom: '✍️ Custom',
      rw_ph: 'E.g.: make it from a girl perspective / emphasize 50% discount...',
      rw_go: 'Rewrite', rw_loading: '⚡ AI is rewriting script...',
      audience: 'Target audience', pains: 'Audience pains', features: 'Killer features', seo_title: 'SEO caption & tags',
      virality: 'virality', script: 'Script',
      tg_title: 'Telegram Ad Posts & ERID',
      tg_lead: 'Create native blogger ad posts with legal ad disclosure (erid) and product photo watermarks.',
      erid_title: 'Ad Disclosure (ERID / Legal)',
      tg_generate: 'Generate 3 Ad Posts',
      wm_title: 'Product Photo Watermark Studio',
      wm_desc: 'Automatically loads the product image (or upload yours) and stamps your @handle and erid token.',
      balance: 'Your balance', credits: '★ Stars balance (1 script = 100 ★, 1 post = 75 ★)', topup: 'Top up with Telegram Stars',
      topup_title: 'Top up packages (Telegram Stars ★)',
      gens: 'Projects created', spent: 'Stars spent ★',
      pack_credits: '★ balance', popular: 'Best Value',
      promo: 'Promo code', promo_ph: 'Enter promo code (e.g. SNIPER)', apply: 'Apply',
      promo_ok: 'Promo applied: +{n} ★', promo_bad: 'Invalid promo code', promo_used: 'Promo code already used',
      history: 'Project history', clear: 'Clear', history_empty: 'Nothing here yet. Generate your first script.',
      transactions: 'Transactions', tx_empty: 'No transactions yet',
      tx_bonus: 'Welcome bonus', tx_topup: 'Telegram Stars Top-up ★', tx_gen: 'Script generation', tx_tg: 'TG Post generation', tx_promo: 'Promo code',
      open: 'Open', plan: 'Creator PRO', guest: 'Creator',
      pay_title: 'Top Up Balance ★', pay_pack: 'Added to balance', pay_price: 'Price',
      pay_note: 'Secure official payment via Telegram Stars (★). Added immediately.',
      pay_stars_btn: '⭐ Pay {s} Stars',
      pay_demo_btn: '⚡ Instant Demo Top-up',
      cancel: 'Cancel', processing: 'Creating invoice...',
      pay_done: 'Balance topped up: +{n} ★!',
      nofunds_title: 'Not enough ★ balance',
      nofunds_text: 'You need {n} ★ (1 script = 100 ★, TG post = 75 ★). Top up via Telegram Stars.',
      err_query: 'Enter a link, SKU or product description',
      err_timeout: 'Server took too long. Please try again.',
      err_network: 'Connection error. Please try again.',
      err_generic: 'Could not process request',
      cleared: 'History cleared'
    }
  };

  // ================= Pricing & State (100 Stars / Script, 75 Stars / Post) =================
  const STORE_KEY = 'sniper_state_v8';
  const COST = 100;      // 100 Stars per Script Funnel
  const TG_COST = 75;    // 75 Stars per Telegram Ad Post
  const WELCOME_BONUS = 100; // 1 free script generation on start
  const PROMO_CODES = { 'SNIPER': 100, 'UGC2026': 150 };
  const PACKS = [
    { id: 'pack_100',  credits: 100,  stars: 100, pop: false },
    { id: 'pack_300',  credits: 300,  stars: 300, pop: true },
    { id: 'pack_1000', credits: 1000, stars: 850, pop: false }
  ];

  function detectLang() {
    const c = ((tgUser && tgUser.language_code) || navigator.language || 'ru').slice(0, 2).toLowerCase();
    return ['ru', 'uk', 'be', 'kk'].indexOf(c) >= 0 ? 'ru' : 'en';
  }
  function defaultState() {
    return {
      lang: detectLang(),
      devMode: checkIsDefaultOwner(),
      balance: WELCOME_BONUS,
      gens: 0,
      spent: 0,
      promoUsed: false,
      history: [],
      tx: [{ type: 'bonus', amount: WELCOME_BONUS, ts: Date.now() }]
    };
  }
  function loadState() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const o = JSON.parse(raw);
        if (o && typeof o === 'object') {
          const st = Object.assign(defaultState(), o);
          if (checkIsDefaultOwner() && o.devMode === undefined) st.devMode = true;
          return st;
        }
      }
    } catch (e) {}
    return defaultState();
  }
  function saveState() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(S));
    } catch (e) {
      try {
        S.history = S.history.slice(0, 5);
        localStorage.setItem(STORE_KEY, JSON.stringify(S));
      } catch (e2) {}
    }
  }
  let S = loadState();
  if (!I18N[S.lang]) S.lang = 'ru';

  function t(key, vars) {
    let s = (I18N[S.lang] && I18N[S.lang][key]) || I18N.en[key] || key;
    if (vars) Object.keys(vars).forEach(k => { s = s.split('{' + k + '}').join(vars[k]); });
    return s;
  }

  // ================= Developer Mode Controls =================
  let secretTapCount = 0;
  let secretTapTimer = null;

  function onBrandSecretTap() {
    secretTapCount++;
    clearTimeout(secretTapTimer);
    if (secretTapCount >= 5) {
      secretTapCount = 0;
      toggleDevMode(!S.devMode);
    } else {
      secretTapTimer = setTimeout(() => { secretTapCount = 0; }, 1400);
    }
  }

  function toggleDevMode(enable) {
    S.devMode = !!enable;
    saveState();
    updateBalanceUI();
    applyLang();
    showToast(S.devMode ? '⚡ DEVELOPER MODE АКТИВИРОВАН!' : 'Developer Mode отключён');
    haptic('success');
  }

  function devAddCredits
