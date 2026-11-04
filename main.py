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
APP_URL_VERSIONED = APP_URL.rstrip("/") + "/?v=7"

# Опционально можно добавить свой числовой Telegram ID в Render -> Environment -> ADMIN_TG_ID
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
        if "flash" not in name or any(w in name for w in _BAD_WORDS):
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
# REQUEST SCHEMAS & PACKAGES
# =====================================================================
class PreviewRequest(BaseModel):
    query: str


class AnalyzeRequest(BaseModel):
    query: str
    format_type: str = "ugc"
    tone: str = "native"
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
    "pack_50": {
        "credits": 50,
        "stars": 25,
        "title_ru": "50 кредитов SNIPER UGC",
        "title_en": "50 Credits SNIPER UGC",
        "desc_ru": "Пакет на 5 генераций сценариев или 10 рекламных TG-постов",
        "desc_en": "Pack for 5 video script funnels or 10 Telegram ad posts",
    },
    "pack_150": {
        "credits": 150,
        "stars": 65,
        "title_ru": "150 кредитов SNIPER UGC (Хит)",
        "title_en": "150 Credits SNIPER UGC (Hot)",
        "desc_ru": "Пакет на 15 генераций сценариев или 30 рекламных TG-постов",
        "desc_en": "Pack for 15 video script funnels or 30 Telegram ad posts",
    },
    "pack_500": {
        "credits": 500,
        "stars": 190,
        "title_ru": "500 кредитов SNIPER UGC PRO",
        "title_en": "500 Credits SNIPER UGC PRO",
        "desc_ru": "PRO-пакет для селлеров, блогеров и агентств",
        "desc_en": "PRO pack for sellers, creators and agencies",
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
# GEMINI GENERATION (SCRIPTS + TG AD POSTS)
# =====================================================================
SYSTEM_INSTRUCTION = """
You are a top producer of viral short-form content for TikTok, Reels, Shorts, and Telegram Ad Posts.
Analyze the product and produce a compact, high-converting promotion funnel.
You ALWAYS reply with STRICTLY valid JSON, no markdown, no ```json fences.
Keep every text field concise and punchy.

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
      "virality_score": 95,
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
        temperature=0.78,
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
# TELEGRAM BOT + TELEGRAM STARS PAYMENTS
# =====================================================================
BOT_TXT = {
    "ru": {
        "start": "<b>SNIPER // UGC STUDIO</b>\n\nОтправь артикул или ссылку на Wildberries / Ozon / сайт, чтобы получить сценарии и рекламные посты, "
                 "или открой обновлённую студию с телесуфлёром и маркировкой рекламы:",
        "open": "⚡ Открыть SNIPER UGC Studio",
        "wait": "🔍 <b>Сканирую карточку товара и собираю сценарии...</b>",
        "hooks": "🔥 <b>ТОП-ХУКИ:</b>",
        "script": "🎬 <b>СЦЕНАРИЙ:",
        "viral": "виральность",
        "hook": "Хук:",
        "full": "⚡ Открыть суфлёр и раскадровку",
        "err": "⚠️ Ошибка:",
        "paid": "✅ <b>Оплата прошла успешно!</b>\nКредиты зачислены на твой баланс в приложении SNIPER UGC.",
    },
    "en": {
        "start": "<b>SNIPER // UGC STUDIO</b>\n\nSend a Wildberries / Ozon / website link or SKU to get ready-to-shoot scripts, "
                 "or open the studio with Teleprompter & Ad Post Builder:",
        "open": "⚡ Open SNIPER UGC Studio",
        "wait": "🔍 <b>Scanning product and building viral funnel...</b>",
        "hooks": "🔥 <b>TOP HOOKS:</b>",
        "script": "🎬 <b>SCRIPT:",
        "viral": "virality",
        "hook": "Hook:",
        "full": "⚡ Open Teleprompter & Storyboard",
        "err": "⚠️ Error:",
        "paid": "✅ <b>Payment successful!</b>\nCredits have been added to your SNIPER UGC balance.",
    },
}


def user_lang(message: types.Message) -> str:
    code = message.from_user.language_code if message.from_user else ""
    return norm_lang(code)


@dp.pre_checkout_query()
async def on_pre_checkout(pre_checkout_query: PreCheckoutQuery):
    await pre_checkout_query.answer(ok=True)


@dp.message(F.successful_payment)
async def on_successful_payment(message: types.Message):
    L = BOT_TXT[user_lang(message)]
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=L["open"], web_app=WebAppInfo(url=APP_URL_VERSIONED))]
    ])
    await message.answer(L["paid"], reply_markup=kb, parse_mode="HTML")


@dp.message(Command("start"))
async def bot_start(message: types.Message):
    L = BOT_TXT[user_lang(message)]
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=L["open"], web_app=WebAppInfo(url=APP_URL_VERSIONED))]
    ])
    await message.answer(L["start"], reply_markup=kb, parse_mode="HTML")


@dp.message()
async def bot_quick_message(message: types.Message):
    txt = (message.text or "").strip()
    if not txt:
        return
    lang = user_lang(message)
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
# FRONTEND UI (MODERN MOBILE STUDIO + DEVELOPER MODE + TELEPROMPTER)
# =====================================================================
FRONTEND_UI = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<title>SNIPER // UGC STUDIO</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Unbounded:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #060609;
    --surface: #0f0f15;
    --surface2: #16161f;
    --surface3: #1e1e2a;
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(212, 255, 58, 0.45);
    --text: #f5f5f7;
    --muted: #8b8b9c;
    --accent: #d4ff3a;
    --accent-glow: rgba(212, 255, 58, 0.22);
    --accent-ink: #08080a;
    --purple: #8b5cf6;
    --danger: #ff5c5c;
    --ok: #34d399;
    --radius: 18px;
    --font: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif;
    --display: 'Unbounded', 'Manrope', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; line-height: 1.45; overflow-x: hidden; }
  body {
    min-height: 100vh;
    background:
      radial-gradient(650px 320px at 95% -5%, rgba(212, 255, 58, 0.09), transparent 60%),
      radial-gradient(550px 320px at -10% 15%, rgba(139, 92, 246, 0.09), transparent 60%),
      var(--bg);
  }

  .app { max-width: 520px; margin: 0 auto; padding: 14px 16px calc(105px + env(safe-area-inset-bottom)); }
  .mono { font-family: var(--mono); }
  button { font-family: inherit; user-select: none; }

  /* TOPBAR & DEVELOPER MODE BADGE */
  .topbar {
    display: flex; justify-content: space-between; align-items: center; gap: 8px;
    margin-bottom: 18px; padding: 6px 0;
  }
  .brand { display: flex; align-items: center; gap: 8px; font-family: var(--display); font-weight: 700; font-size: 14px; letter-spacing: 0.4px; cursor: pointer; }
  .brand .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 14px var(--accent); animation: pulseDot 2.4s infinite; }
  @keyframes pulseDot { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.25); opacity: 0.75; } }
  .brand .sub { color: var(--muted); font-weight: 500; font-size: 10.5px; background: var(--surface2); padding: 2px 7px; border-radius: 6px; border: 1px solid var(--border); }

  .dev-top-badge {
    display: none;
    align-items: center; gap: 4px;
    font-family: var(--mono); font-size: 9.5px; font-weight: 800; letter-spacing: 0.5px;
    color: #000; background: linear-gradient(135deg, #d4ff3a, #34d399);
    padding: 3px 8px; border-radius: 6px;
    box-shadow: 0 0 16px rgba(212, 255, 58, 0.45);
    text-transform: uppercase;
  }
  .dev-top-badge.active { display: inline-flex; }

  .top-actions { display: flex; align-items: center; gap: 8px; }
  .bal-chip {
    display: flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(212,255,58,0.12), rgba(255,255,255,0.03));
    border: 1px solid rgba(212,255,58,0.3); color: var(--text);
    padding: 7px 12px; border-radius: 999px; font-weight: 800; font-size: 13px; cursor: pointer;
    transition: transform 0.15s ease;
  }
  .bal-chip:active { transform: scale(0.95); }
  .bal-chip .coin { color: var(--accent); }
  .lang-sw { display: flex; background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 3px; }
  .lang-sw button {
    background: transparent; border: none; color: var(--muted); font-weight: 700; font-size: 11px;
    padding: 5px 9px; border-radius: 999px; cursor: pointer; transition: all 0.2s ease;
  }
  .lang-sw button.active { background: var(--text); color: #000; }

  /* DEVELOPER CONSOLE CARD IN CABINET */
  .dev-console {
    display: none;
    background: linear-gradient(135deg, rgba(212, 255, 58, 0.12), rgba(52, 211, 153, 0.08)), #0b0b10;
    border: 1.5px solid var(--accent);
    border-radius: var(--radius);
    padding: 15px; margin-bottom: 14px;
    box-shadow: 0 0 28px rgba(212, 255, 58, 0.14);
  }
  .dev-console.active { display: block; animation: viewEnter 0.25s ease; }
  .dev-console-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid rgba(212, 255, 58, 0.2);
  }
  .dev-console-title {
    font-family: var(--mono); font-size: 11.5px; font-weight: 800; color: var(--accent);
    letter-spacing: 0.6px; display: flex; align-items: center; gap: 6px;
  }
  .dev-actions-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }

  /* VIEWS */
  .view { display: none; }
  .view.active { display: block; animation: viewEnter 0.3s cubic-bezier(0.16, 1, 0.3, 1); }
  @keyframes viewEnter { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  h1.title { font-family: var(--display); font-size: 19px; font-weight: 700; line-height: 1.25; margin-bottom: 6px; letter-spacing: -0.02em; }
  p.lead { color: var(--muted); font-size: 13px; margin-bottom: 16px; }

  .panel, .card {
    background: rgba(15, 15, 21, 0.85); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 15px; margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
  }
  .lbl { font-family: var(--mono); font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }

  /* SLEEK MOBILE SMART INPUT POD */
  .smart-input-pod {
    background: var(--bg);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 10px 12px;
    margin-bottom: 12px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .smart-input-pod:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  .smart-input-row { display: flex; align-items: center; gap: 8px; }
  .smart-icon {
    width: 34px; height: 34px; border-radius: 10px;
    background: var(--surface2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    color: var(--accent); flex-shrink: 0; font-size: 16px;
  }
  .smart-field {
    flex: 1; min-width: 0; background: transparent; border: none; outline: none;
    color: var(--text); font-size: 14px; font-family: inherit; font-weight: 500;
    line-height: 1.35; resize: none; max-height: 90px; overflow-y: auto;
  }
  .smart-field::placeholder { color: #565666; font-weight: 400; }
  .smart-actions { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .paste-pill {
    background: var(--surface2); color: var(--text); border: 1px solid var(--border);
    padding: 6px 10px; border-radius: 10px; font-size: 11px; font-weight: 700;
    cursor: pointer; display: flex; align-items: center; gap: 4px; transition: all 0.15s ease;
  }
  .paste-pill:active { transform: scale(0.94); background: var(--accent); color: #000; }
  .clear-pill {
    width: 26px; height: 26px; border-radius: 8px; background: var(--surface2);
    color: var(--muted); border: 1px solid var(--border); display: none;
    align-items: center; justify-content: center; font-size: 12px; cursor: pointer;
  }

  /* LIVE PRODUCT OVERVIEW PREVIEW CARD */
  .live-preview-wrap {
    display: none; margin-bottom: 14px;
    background: linear-gradient(135deg, rgba(212, 255, 58, 0.08), rgba(139, 92, 246, 0.07)), var(--surface2);
    border: 1px solid var(--border-strong); border-radius: 14px; padding: 11px 12px;
    animation: slideDown 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes slideDown { from { opacity: 0; transform: translateY(-8px) scale(0.98); } to { opacity: 1; transform: none; } }
  .lp-inner { display: flex; align-items: center; gap: 12px; }
  .lp-thumb {
    width: 54px; height: 54px; border-radius: 10px; object-fit: cover;
    background: #0b0b0f; border: 1px solid var(--border); flex-shrink: 0;
  }
  .lp-body { flex: 1; min-width: 0; }
  .lp-badges { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap; }
  .lp-badge {
    font-family: var(--mono); font-size: 9.5px; font-weight: 700; text-transform: uppercase;
    padding: 2px 7px; border-radius: 6px; background: var(--accent); color: #000;
  }
  .lp-price {
    font-family: var(--mono); font-size: 10.5px; font-weight: 700;
    padding: 2px 7px; border-radius: 6px; background: rgba(52, 211, 153, 0.16); color: var(--ok);
  }
  .lp-title { font-weight: 700; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text); }
  .lp-sub { font-size: 11px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 2px; }

  /* CHIPS & BUTTONS */
  .chips { display: flex; gap: 6px; overflow-x: auto; margin-bottom: 14px; scrollbar-width: none; padding-bottom: 2px; }
  .chips::-webkit-scrollbar { display: none; }
  .chip {
    flex-shrink: 0; background: var(--bg); color: var(--muted); border: 1px solid var(--border);
    border-radius: 999px; padding: 8px 13px; font-size: 12px; font-weight: 600; cursor: pointer;
    transition: all 0.18s ease;
  }
  .chip.active { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); box-shadow: 0 4px 14px var(--accent-glow); }

  .btn {
    width: 100%; background: var(--accent); color: var(--accent-ink); border: none; padding: 14px 16px;
    font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.4px;
    border-radius: 14px; cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 8px;
    box-shadow: 0 6px 20px var(--accent-glow); transition: transform 0.12s ease, opacity 0.15s ease;
  }
  .btn:active { transform: scale(0.98); }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
  .btn.ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border); box-shadow: none; }
  .btn.small { width: auto; padding: 9px 14px; font-size: 11.5px; border-radius: 10px; box-shadow: none; }
  .cost-note { text-align: center; color: var(--muted); font-size: 11px; margin-top: 9px; font-family: var(--mono); }

  .err-box {
    display: none; background: rgba(255,92,92,0.1); border: 1px solid rgba(255,92,92,0.4); color: #ffb3b3;
    border-radius: 14px; padding: 12px; margin-bottom: 14px; font-size: 13px;
  }

  /* AI LIVE TERMINAL LOADER */
  .ai-terminal {
    display: none; background: #09090d; border: 1px solid var(--border-strong);
    border-radius: var(--radius); padding: 16px; margin-bottom: 16px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.5), 0 0 25px rgba(212, 255, 58, 0.08);
  }
  .term-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .term-dots { display: flex; gap: 6px; align-items: center; }
  .term-dots span { width: 8px; height: 8px; border-radius: 50%; background: var(--border); }
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
  .term-line { margin-bottom: 5px; display: flex; gap: 8px; align-items: flex-start; animation: fade 0.2s ease; }
  .term-line .ok { color: var(--accent); flex-shrink: 0; }
  .term-cursor { display: inline-block; width: 7px; height: 13px; background: var(--accent); vertical-align: middle; margin-left: 4px; animation: blink 0.8s infinite; }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

  /* RESULTS */
  #results { display: none; }
  .prod-head { display: flex; gap: 12px; align-items: center; padding: 13px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 14px; }
  .prod-thumb { width: 58px; height: 58px; border-radius: 12px; object-fit: cover; background: #111; flex-shrink: 0; border: 1px solid var(--border); }
  .tag { font-family: var(--mono); font-size: 10px; border: 1px solid var(--border); padding: 3px 8px; border-radius: 999px; color: var(--muted); background: var(--surface2); }

  .tabs {
    display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none;
    background: var(--surface); padding: 4px; border: 1px solid var(--border); border-radius: 14px; margin-bottom: 14px;
  }
  .tabs::-webkit-scrollbar { display: none; }
  .tab-btn {
    flex: 1; white-space: nowrap; background: transparent; color: var(--muted); border: none;
    padding: 9px 10px; font-size: 10.5px; font-weight: 800; text-transform: uppercase; border-radius: 10px; cursor: pointer;
    transition: all 0.15s ease;
  }
  .tab-btn.active { background: var(--accent); color: #000; }
  .tab-content { display: none; }
  .tab-content.active { display: block; animation: viewEnter 0.25s ease; }

  .card-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border); flex-wrap: wrap; }
  .action-row { display: flex; gap: 6px; align-items: center; }
  .copy-btn {
    background: var(--surface2); border: 1px solid var(--border); color: var(--text);
    padding: 6px 10px; border-radius: 9px; font-size: 10.5px; font-weight: 700; cursor: pointer;
    font-family: var(--mono); display: inline-flex; align-items: center; gap: 4px;
  }
  .copy-btn:active { background: var(--accent); color: #000; }
  .prompter-btn {
    background: linear-gradient(135deg, var(--accent), #a3e635); color: #000; border: none;
    padding: 6px 11px; border-radius: 9px; font-size: 10.5px; font-weight: 800; cursor: pointer;
    font-family: var(--mono); display: inline-flex; align-items: center; gap: 5px;
    box-shadow: 0 3px 12px var(--accent-glow);
  }
  .prompter-btn:active { transform: scale(0.95); }

  .hook-box { background: var(--bg); border-left: 3px solid var(--accent); padding: 10px 12px; font-size: 13px; margin-bottom: 10px; border-radius: 0 10px 10px 0; }
  .scene-item { background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 11px 12px; margin-bottom: 8px; font-size: 12.5px; }
  .prompt-box { margin-top: 8px; padding: 8px 10px; background: var(--surface); border: 1px dashed var(--border); border-radius: 10px; font-size: 11px; color: #b0b0bd; display: flex; justify-content: space-between; align-items: center; gap: 8px; }

  /* TG POST & ERID & WATERMARK STUDIO */
  .input-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
  input.inp {
    width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 11px;
    color: var(--text); padding: 10px 12px; font-size: 13px; font-family: inherit; outline: none;
  }
  input.inp:focus { border-color: var(--accent); }
  .erid-preview {
    background: rgba(139, 92, 246, 0.08); border: 1px dashed rgba(139, 92, 246, 0.45);
    border-radius: 11px; padding: 9px 11px; font-size: 11.5px; color: #d8b4fe; margin-top: 8px;
    font-family: var(--mono);
  }
  .tg-post-body { white-space: pre-line; font-size: 13.5px; line-height: 1.5; margin: 8px 0 10px; color: #ececf3; }
  .wm-canvas-wrap {
    width: 100%; background: #050508; border: 1px solid var(--border); border-radius: 14px;
    overflow: hidden; margin: 10px 0; display: flex; align-items: center; justify-content: center; min-height: 220px;
  }
  canvas#wmCanvas { max-width: 100%; max-height: 340px; display: block; border-radius: 10px; }

  /* FULLSCREEN TELEPROMPTER OVERLAY */
  .teleprompter-modal {
    position: fixed; inset: 0; z-index: 500; background: #000000;
    display: none; flex-direction: column; justify-content: space-between;
    padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom);
  }
  .teleprompter-modal.open { display: flex; animation: fade 0.2s ease; }
  .tp-topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; background: rgba(15, 15, 21, 0.9); border-bottom: 1px solid var(--border); z-index: 5;
  }
  .tp-title { font-family: var(--mono); font-size: 12px; color: var(--accent); font-weight: 700; display: flex; align-items: center; gap: 6px; }
  .tp-stage {
    position: relative; flex: 1; overflow-y: auto; padding: 38vh 22px 45vh;
    scroll-behavior: auto;
  }
  .tp-stage.mirror { transform: scaleX(-1); }
  .tp-eye-guide {
    position: fixed; top: 36%; left: 0; right: 0; height: 90px; pointer-events: none; z-index: 3;
    border-top: 1.5px dashed rgba(212, 255, 58, 0.45);
    border-bottom: 1.5px dashed rgba(212, 255, 58, 0.45);
    background: linear-gradient(90deg, rgba(212,255,58,0.06), transparent 15%, transparent 85%, rgba(212,255,58,0.06));
  }
  .tp-eye-tag {
    position: fixed; top: calc(36% - 18px); left: 16px; z-index: 4; pointer-events: none;
    font-family: var(--mono); font-size: 9px; color: var(--accent); text-transform: uppercase; letter-spacing: 1px;
  }
  .tp-text {
    font-size: 28px; font-weight: 800; line-height: 1.45; color: #ffffff;
    text-align: center; outline: none; white-space: pre-wrap;
  }
  .tp-countdown {
    position: fixed; inset: 0; z-index: 10; background: rgba(0,0,0,0.82);
    display: none; align-items: center; justify-content: center;
    font-family: var(--display); font-size: 86px; font-weight: 700; color: var(--accent);
  }
  .tp-controls {
    background: rgba(15, 15, 21, 0.95); border-top: 1px solid var(--border);
    padding: 12px 16px 16px; z-index: 5;
  }
  .tp-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
  .tp-row:last-child { margin-bottom: 0; }
  .tp-slider { flex: 1; accent-color: var(--accent); }

  /* CABINET & STARS */
  .profile { display: flex; align-items: center; gap: 12px; }
  .avatar { width: 52px; height: 52px; border-radius: 50%; background: var(--surface2); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-family: var(--display); font-size: 18px; font-weight: 700; color: var(--accent); overflow: hidden; flex-shrink: 0; }
  .avatar img { width: 100%; height: 100%; object-fit: cover; }
  .balance-card {
    background: linear-gradient(135deg, rgba(212,255,58,0.15), rgba(139,92,246,0.12)), var(--surface);
    border: 1px solid var(--border-strong); border-radius: 20px; padding: 18px; margin-bottom: 14px;
  }
  .bal-num { font-family: var(--display); font-size: 36px; font-weight: 700; line-height: 1.1; margin: 6px 0 2px; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }
  .stat .v { font-family: var(--display); font-size: 20px; font-weight: 600; }
  .stat .k { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .sec-title { font-family: var(--display); font-size: 13px; font-weight: 600; margin: 18px 0 10px; }
  .packs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
  .pack { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 13px 8px; text-align: center; cursor: pointer; position: relative; color: var(--text); transition: transform 0.15s ease; }
  .pack:active { transform: scale(0.96); }
  .pack.pop { border-color: var(--accent); background: linear-gradient(180deg, rgba(212,255,58,0.08), var(--surface)); }
  .pack .pc { font-family: var(--display); font-size: 19px; font-weight: 700; }
  .pack .pk { color: var(--muted); font-size: 10px; margin-bottom: 8px; }
  .pack .pp { font-weight: 800; font-size: 13px; color: var(--accent); }
  .pack .badge { position: absolute; top: -8px; left: 50%; transform: translateX(-50%); background: var(--accent); color: #000; font-size: 9px; font-weight: 800; padding: 2px 7px; border-radius: 999px; text-transform: uppercase; }
  .list-item { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--border); }
  .list-item:last-child { border-bottom: none; }
  .li-main { min-width: 0; }
  .li-title { font-weight: 700; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .li-sub { color: var(--muted); font-size: 11px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .plus { color: var(--ok); font-weight: 800; }
  .minus { color: var(--danger); font-weight: 800; }
  .empty { color: var(--muted); font-size: 12px; text-align: center; padding: 14px 0; }

  /* BOTTOM NAV (3 TABS) */
  .nav {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 50;
    padding: 8px 14px calc(8px + env(safe-area-inset-bottom));
    background: rgba(6, 6, 9, 0.9); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-top: 1px solid var(--border);
  }
  .nav-inner { max-width: 520px; margin: 0 auto; display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  .nav button {
    background: transparent; border: none; color: var(--muted); display: flex; flex-direction: column; align-items: center; gap: 4px;
    font-size: 10.5px; font-weight: 700; padding: 6px; border-radius: 12px; cursor: pointer; transition: color 0.15s ease;
  }
  .nav button.active { color: var(--accent); }
  .nav svg { width: 21px; height: 21px; stroke: currentColor; fill: none; stroke-width: 1.9; stroke-linecap: round; stroke-linejoin: round; }

  /* SHEET MODAL */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.68); backdrop-filter: blur(5px); z-index: 100; display: none; align-items: flex-end; justify-content: center; }
  .overlay.open { display: flex; }
  .sheet { width: 100%; max-width: 520px; background: var(--surface); border: 1px solid var(--border); border-radius: 22px 22px 0 0; padding: 20px 18px calc(22px + env(safe-area-inset-bottom)); animation: up 0.22s ease; }
  @keyframes up { from { transform: translateY(30px); opacity: 0; } to { transform: none; opacity: 1; } }
  .sheet h3 { font-family: var(--display); font-size: 16px; font-weight: 600; margin-bottom: 12px; }
  .sheet .kv { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
  .sheet p { color: var(--muted); font-size: 12px; margin: 10px 0 14px; }
  .sheet .btn + .btn { margin-top: 8px; }

  .toast {
    position: fixed; bottom: calc(88px + env(safe-area-inset-bottom)); left: 50%; transform: translateX(-50%);
    background: var(--accent); color: #000; padding: 9px 18px; border-radius: 999px; font-size: 12px; font-weight: 800;
    display: none; z-index: 600; white-space: nowrap; box-shadow: 0 6px 20px rgba(0,0,0,0.4);
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
      <span id="normalStudioSub" class="sub">STUDIO 4.0</span>
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

  <!-- ==================== VIEW 1: STUDIO (VIDEO SCRIPTS) ==================== -->
  <section id="view-studio" class="view active">
    <h1 class="title" data-i18n="studio_title"></h1>
    <p class="lead" data-i18n="studio_lead"></p>

    <div class="panel">
      <div class="lbl">
        <span data-i18n="src_label"></span>
        <span class="mono" style="color:var(--accent);" id="detectBadge">WB • OZON • WEB</span>
      </div>

      <!-- MODERN MOBILE SMART INPUT -->
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
        <span>⚡</span><span data-i18n="generate"></span><span>→</span>
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
      <div class="prod-head" id="pHeader"></div>
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
    <h1 class="title" data-i18n="tg_title"></h1>
    <p class="lead" data-i18n="tg_lead"></p>

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
        <span>✨</span><span data-i18n="tg_generate"></span><span id="tgCostLabel">(5 ★)</span>
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

    <!-- PERSONAL DEVELOPER CONSOLE (VISIBLE ONLY TO OWNER) -->
    <div id="devConsoleCard" class="dev-console">
      <div class="dev-console-head">
        <div class="dev-console-title"><span>⚡</span><span>DEVELOPER MODE // ROOT ACCESS</span></div>
        <span class="tag" style="background:var(--accent); color:#000; font-weight:800; border:none;">ACTIVE</span>
      </div>
      <div style="font-size:12px; color:#d1d1e0; line-height:1.45;">
        Персональный режим владельца активирован. Все генерации сценариев и TG-постов работают <strong>бесплатно и без ограничений (∞)</strong>.
      </div>
      <div class="dev-actions-grid">
        <button class="btn small" onclick="devAddCredits(1000)">+1000 КРЕДИТОВ</button>
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

<!-- BOTTOM NAVIGATION (3 ITEMS) -->
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
      if (tg.setHeaderColor) tg.setHeaderColor('#060609');
      if (tg.setBackgroundColor) tg.setBackgroundColor('#060609');
    } catch (e) {}
  }
  const tgUser = (tg && tg.initDataUnsafe && tg.initDataUnsafe.user) ? tg.initDataUnsafe.user : null;

  // ================= Auto-Detect Owner (Developer Mode) =================
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
      studio_title: 'Виральные ролики и суфлёр',
      studio_lead: 'Вставь ссылку или артикул WB / Ozon — сразу увидишь карточку товара, получишь сценарии, суфлёр и посты.',
      src_label: 'Товар (ссылка или артикул)',
      src_ph: 'Вставь ссылку WB, Ozon или артикул...',
      paste: 'Вставить',
      fmt_label: 'Формат ролика', tone_label: 'Тональность подачи',
      f_ugc: 'UGC обзор', f_ps: 'Боль → Решение', f_unb: 'Распаковка / ASMR', f_aes: 'Aesthetic', f_story: 'Сторителлинг',
      t_native: 'Нативный', t_aggr: 'Дерзкий', t_humor: 'Юмор', t_lux: 'Премиум',
      generate: 'Собрать сценарии и посты', cost: 'Стоимость полного разбора: {n} кр.',
      cost_dev: '⚡ DEVELOPER MODE: БЕСПЛАТНО (БЕЗЛИМИТ ∞)',
      tab_scripts: 'Сценарии', tab_tgposts: 'TG Посты', tab_hooks: 'Хуки', tab_video: 'AI Видео', tab_seo: 'ЦА & SEO',
      frame: 'Кадр', voice: 'Озвучка', screen_text: 'Текст', hook: 'ХУК', cta: 'CTA',
      copy: 'КОПИРОВАТЬ', copy_all: 'ВЕСЬ СЦЕНАРИЙ', copy_seo: 'КОПИРОВАТЬ SEO', copied: 'Скопировано в буфер!',
      prompter: '🎬 СУФЛЁР',
      audience: 'Целевая аудитория', pains: 'Боли ЦА', features: 'Главные фишки', seo_title: 'SEO описание и хэштеги',
      virality: 'виральность', script: 'Сценарий',
      tg_title: 'Рекламные TG-посты & Маркировка',
      tg_lead: 'Генерация нативных постов для блогеров с автоматической маркировкой рекламы (erid) и водяным знаком на фото.',
      erid_title: 'Маркировка рекламы (ОРД / ERID)',
      tg_generate: 'Сгенерировать 3 рекламных поста',
      wm_title: 'Водяной знак и ERID на фото товара',
      wm_desc: 'Автоматически берёт фото спарсенного товара (или загрузи своё) и накладывает твой ник и токен erid.',
      balance: 'Твой баланс', credits: 'кредитов для генераций', topup: 'Пополнить через Telegram Stars',
      topup_title: 'Пополнение баланса (Telegram Stars ★)',
      gens: 'Генераций', spent: 'Потрачено кр.',
      pack_credits: 'кредитов', popular: 'Хит',
      promo: 'Промокод', promo_ph: 'Введи промокод (напр. SNIPER)', apply: 'Применить',
      promo_ok: 'Промокод активирован: +{n} кр.', promo_bad: 'Неверный промокод', promo_used: 'Промокод уже использован',
      history: 'История проектов', clear: 'Очистить', history_empty: 'Пока пусто. Сгенерируй свой первый проект.',
      transactions: 'История операций', tx_empty: 'Операций пока нет',
      tx_bonus: 'Приветственный бонус', tx_topup: 'Пополнение Stars ★', tx_gen: 'Генерация проекта', tx_promo: 'Промокод',
      open: 'Открыть', plan: 'Creator PRO', guest: 'Креатор',
      pay_title: 'Пополнение баланса', pay_pack: 'Пакет', pay_price: 'К оплате',
      pay_note: 'Оплата проходит официально через Telegram Stars (★). Кредиты начисляются моментально после подтверждения.',
      pay_stars_btn: '⭐ Оплатить {s} Stars',
      pay_demo_btn: '⚡ Тестовое пополнение (Демо)',
      cancel: 'Отмена', processing: 'Создаём счёт...',
      pay_done: 'Баланс пополнен: +{n} кр.!',
      nofunds_title: 'Недостаточно кредитов',
      nofunds_text: 'Для генерации нужно {n} кр. Пополни баланс через Telegram Stars.',
      err_query: 'Вставь ссылку, артикул или описание товара',
      err_timeout: 'Сервер отвечает слишком долго. Попробуй ещё раз.',
      err_network: 'Ошибка соединения. Попробуй ещё раз.',
      err_generic: 'Не удалось обработать запрос',
      cleared: 'История очищена'
    },
    en: {
      nav_studio: 'Scripts', nav_tgads: 'TG Ads & ERID', nav_cabinet: 'Account',
      studio_title: 'Viral Scripts & Teleprompter',
      studio_lead: 'Paste a WB / Ozon / website link — get instant product overview, scripts, teleprompter and ad posts.',
      src_label: 'Product (link or SKU)',
      src_ph: 'Paste WB, Ozon or website link / SKU...',
      paste: 'Paste',
      fmt_label: 'Video format', tone_label: 'Tone',
      f_ugc: 'UGC review', f_ps: 'Pain → Solution', f_unb: 'Unboxing / ASMR', f_aes: 'Aesthetic', f_story: 'Storytelling',
      t_native: 'Native', t_aggr: 'Bold', t_humor: 'Humor', t_lux: 'Premium',
      generate: 'Generate Scripts & Posts', cost: 'Full funnel cost: {n} cr.',
      cost_dev: '⚡ DEVELOPER MODE: FREE (UNLIMITED ∞)',
      tab_scripts: 'Scripts', tab_tgposts: 'TG Posts', tab_hooks: 'Hooks', tab_video: 'AI Video', tab_seo: 'Audience & SEO',
      frame: 'Frame', voice: 'Voice', screen_text: 'Text', hook: 'HOOK', cta: 'CTA',
      copy: 'COPY', copy_all: 'FULL SCRIPT', copy_seo: 'COPY SEO', copied: 'Copied to clipboard!',
      prompter: '🎬 PROMPTER',
      audience: 'Target audience', pains: 'Audience pains', features: 'Killer features', seo_title: 'SEO caption & tags',
      virality: 'virality', script: 'Script',
      tg_title: 'Telegram Ad Posts & ERID',
      tg_lead: 'Create native blogger ad posts with legal ad disclosure (erid) and product photo watermarks.',
      erid_title: 'Ad Disclosure (ERID / Legal)',
      tg_generate: 'Generate 3 Telegram Ad Posts',
      wm_title: 'Product Photo Watermark Studio',
      wm_desc: 'Automatically loads the product image (or upload yours) and stamps your @handle and erid token.',
      balance: 'Your balance', credits: 'generation credits', topup: 'Top up with Telegram Stars',
      topup_title: 'Top up balance (Telegram Stars ★)',
      gens: 'Generations', spent: 'Credits spent',
      pack_credits: 'credits', popular: 'Hot',
      promo: 'Promo code', promo_ph: 'Enter promo code (e.g. SNIPER)', apply: 'Apply',
      promo_ok: 'Promo applied: +{n} cr.', promo_bad: 'Invalid promo code', promo_used: 'Promo code already used',
      history: 'Project history', clear: 'Clear', history_empty: 'Nothing here yet. Generate your first project.',
      transactions: 'Transactions', tx_empty: 'No transactions yet',
      tx_bonus: 'Welcome bonus', tx_topup: 'Telegram Stars Top-up ★', tx_gen: 'Project generation', tx_promo: 'Promo code',
      open: 'Open', plan: 'Creator PRO', guest: 'Creator',
      pay_title: 'Top Up Balance', pay_pack: 'Package', pay_price: 'Price',
      pay_note: 'Secure official payment via Telegram Stars (★). Credits are added immediately.',
      pay_stars_btn: '⭐ Pay {s} Stars',
      pay_demo_btn: '⚡ Instant Demo Top-up',
      cancel: 'Cancel', processing: 'Creating invoice...',
      pay_done: 'Balance topped up: +{n} cr.!',
      nofunds_title: 'Not enough credits',
      nofunds_text: 'You need {n} cr. Top up your balance via Telegram Stars.',
      err_query: 'Enter a link, SKU or product description',
      err_timeout: 'Server took too long. Please try again.',
      err_network: 'Connection error. Please try again.',
      err_generic: 'Could not process request',
      cleared: 'History cleared'
    }
  };

  // ================= State =================
  const STORE_KEY = 'sniper_state_v6';
  const COST = 10;
  const TG_COST = 5;
  const WELCOME_BONUS = 100;
  const PROMO_CODES = { 'SNIPER': 50, 'UGC2026': 100 };
  const PACKS = [
    { id: 'pack_50',  credits: 50,  stars: 25,  pop: false },
    { id: 'pack_150', credits: 150, stars: 65,  pop: true },
    { id: 'pack_500', credits: 500, stars: 190, pop: false }
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

  function devAddCredits(amount) {
    S.balance += amount;
    addTx('topup', amount);
    saveState();
    updateBalanceUI();
    renderCabinet();
    showToast(`+${amount} кр. начислено (DEV)!`);
    haptic('success');
  }

  // ================= Helpers =================
  let curFmt = 'ugc';
  let curTone = 'native';
  let lastResult = null;
  let currentProductPreview = null;
  const copyStore = [];
  const prompterStore = [];

  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function reg(txt) { copyStore.push(txt == null ? '' : String(txt)); return copyStore.length - 1; }
  function copyIdx(i) { copy(copyStore[i]); }
  function regTp(txt) { prompterStore.push(txt == null ? '' : String(txt)); return prompterStore.length - 1; }

  let toastTimer = null;
  function showToast(text) {
    const el = document.getElementById('toast');
    el.textContent = text;
    el.style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.style.display = 'none'; }, 1900);
  }
  function haptic(kind) {
    try { if (tg && tg.HapticFeedback) tg.HapticFeedback.notificationOccurred(kind || 'success'); } catch (e) {}
  }
  function fallbackCopy(txt) {
    const ta = document.createElement('textarea');
    ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
  }
  function copy(txt) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).catch(() => fallbackCopy(txt));
    } else {
      fallbackCopy(txt);
    }
    showToast(t('copied'));
    haptic('success');
  }
  function fmtDate(ts) {
    try {
      return new Date(ts).toLocaleString(S.lang === 'ru' ? 'ru-RU' : 'en-US', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' });
    } catch (e) { return ''; }
  }

  // ================= Smart Input & Live Product Overview =================
  let previewTimers = {};
  function onSmartInput(el, mode) {
    const val = (el.value || '').trim();
    const clearBtn = document.getElementById(mode === 'main' ? 'clearMainBtn' : 'clearTgBtn');
    if (clearBtn) clearBtn.style.display = val.length > 0 ? 'flex' : 'none';

    const otherId = mode === 'main' ? 'tgQueryInp' : 'queryInp';
    const otherEl = document.getElementById(otherId);
    if (otherEl && !otherEl.value.trim()) otherEl.value = val;

    clearTimeout(previewTimers[mode]);
    const box = document.getElementById(mode === 'main' ? 'livePreviewMain' : 'livePreviewTg');
    if (val.length < 5) {
      if (box) box.style.display = 'none';
      return;
    }

    previewTimers[mode] = setTimeout(() => fetchLivePreview(val, mode), 420);
  }

  async function pasteClipboard(inputId, mode) {
    const el = document.getElementById(inputId);
    try {
      if (navigator.clipboard && navigator.clipboard.readText) {
        const txt = await navigator.clipboard.readText();
        if (txt) {
          el.value = txt.trim();
          onSmartInput(el, mode);
          haptic('success');
          return;
        }
      }
    } catch (e) {}
    el.focus();
  }

  function clearSmartInput(inputId, mode) {
    const el = document.getElementById(inputId);
    el.value = '';
    onSmartInput(el, mode);
    const box = document.getElementById(mode === 'main' ? 'livePreviewMain' : 'livePreviewTg');
    if (box) box.style.display = 'none';
  }

  async function fetchLivePreview(query, mode) {
    const box = document.getElementById(mode === 'main' ? 'livePreviewMain' : 'livePreviewTg');
    if (!box) return;
    try {
      const resp = await fetch('/api/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
      });
      if (!resp.ok) return;
      const d = await resp.json();
      currentProductPreview = d;

      const img = d.image ? `<img class="lp-thumb" src="${esc(d.image)}" alt="" onerror="this.style.display='none'">` : '';
      const price = d.price ? `<span class="lp-price">${esc(d.price)}</span>` : '';
      const brand = d.brand ? ` · ${esc(d.brand)}` : '';
      const cat = d.category ? `${esc(d.category)}` : 'Готово к генерации';

      box.innerHTML = `
        <div class="lp-inner">
          ${img}
          <div class="lp-body">
            <div class="lp-badges">
              <span class="lp-badge">✓ ${esc(d.source || 'НАЙДЕНО')}</span>
              ${price}
            </div>
            <div class="lp-title">${esc(d.title)}</div>
            <div class="lp-sub">${cat}${brand}</div>
          </div>
        </div>
      `;
      box.style.display = 'block';
      if (d.image) loadWatermarkProductImage(d.image);
    } catch (e) {}
  }

  // ================= AI Live Terminal Animation =================
  let termInterval = null;
  function startAiTerminal(queryText) {
    const term = document.getElementById('aiTerminal');
    const bar = document.getElementById('termBar');
    const pct = document.getElementById('termPercent');
    const stream = document.getElementById('termStream');
    term.style.display = 'block';
    stream.innerHTML = '';

    const pTitle = (currentProductPreview && currentProductPreview.title) ? currentProductPreview.title : queryText.slice(0, 42);
    const pSource = (currentProductPreview && currentProductPreview.source) ? currentProductPreview.source : 'Web Source';
    const pPrice = (currentProductPreview && currentProductPreview.price) ? ` (${currentProductPreview.price})` : '';

    const logSteps = S.lang === 'ru' ? [
      { p: 15, txt: `Подключение к ${pSource}: «${pTitle}»${pPrice}` },
      { p: 32, txt: `Извлечение характеристик, УТП и болей целевой аудитории...` },
      { p: 52, txt: `AI-продюсер: генерация 5 триггерных хуков (первые 2 секунды)...` },
      { p: 71, txt: `Сборка покадрового сценария, озвучки для суфлёра и B-Roll промптов...` },
      { p: 88, txt: `Копирайтинг 3 нативных рекламных постов для Telegram...` },
      { p: 96, txt: `Финальная упаковка JSON-проекта...` }
    ] : [
      { p: 15, txt: `Connected to ${pSource}: "${pTitle}"${pPrice}` },
      { p: 32, txt: `Extracting product specs, USP and audience pain points...` },
      { p: 52, txt: `AI Producer: crafting 5 scroll-stopping hooks (0-2s)...` },
      { p: 71, txt: `Writing scene-by-scene storyboard, teleprompter voiceover & AI prompts...` },
      { p: 88, txt: `Drafting 3 native Telegram ad posts...` },
      { p: 96, txt: `Finalizing project package...` }
    ];

    let idx = 0;
    function pushStep() {
      if (idx >= logSteps.length) return;
      const item = logSteps[idx];
      bar.style.width = item.p + '%';
      pct.textContent = `AI ENGINE // ${item.p}%`;

      const prevCursor = stream.querySelector('.term-cursor');
      if (prevCursor) prevCursor.remove();

      const line = document.createElement('div');
      line.className = 'term-line';
      line.innerHTML = `<span class="ok">▸</span><span>${esc(item.txt)}<span class="term-cursor"></span></span>`;
      stream.appendChild(line);
      stream.scrollTop = stream.scrollHeight;
      idx++;
    }

    pushStep();
    clearInterval(termInterval);
    termInterval = setInterval(pushStep, 2300);
  }

  function stopAiTerminal() {
    clearInterval(termInterval);
    const term = document.getElementById('aiTerminal');
    if (term) term.style.display = 'none';
  }

  // ================= Views & Tabs =================
  function showView(name) {
    ['studio', 'tgads', 'cabinet'].forEach(v => {
      document.getElementById('view-' + v).classList.toggle('active', v === name);
      document.getElementById('nav-' + v).classList.toggle('active', v === name);
    });
    if (name === 'cabinet') renderCabinet();
    if (name === 'tgads') drawWatermark();
    window.scrollTo(0, 0);
  }
  function setTab(name, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
  }
  function bindChips(id, cb) {
    document.querySelectorAll('#' + id + ' .chip').forEach(c => {
      c.onclick = () => {
        document.querySelectorAll('#' + id + ' .chip').forEach(x => x.classList.remove('active'));
        c.classList.add('active');
        cb(c.dataset.v);
      };
    });
  }
  bindChips('fmtChips', v => { curFmt = v; });
  bindChips('toneChips', v => { curTone = v; });

  function setLang(l) {
    if (!I18N[l]) return;
    S.lang = l;
    saveState();
    applyLang();
  }
  function applyLang() {
    document.documentElement.lang = S.lang;
    document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
    document.querySelectorAll('.lang-sw button').forEach(b => b.classList.toggle('active', b.dataset.l === S.lang));
    document.getElementById('costNote').textContent = S.devMode ? t('cost_dev') : t('cost', { n: COST });
    const tgCostLbl = document.getElementById('tgCostLabel');
    if (tgCostLbl) tgCostLbl.textContent = S.devMode ? '(DEV ∞)' : '(5 ★)';
    renderCabinet();
    if (lastResult) renderResult(lastResult);
  }

  // ================= Teleprompter Studio =================
  let tpPlaying = false;
  let tpSpeed = 1.2;
  let tpFontSize = 28;
  let tpMirror = false;
  let tpRaf = null;
  let tpLastTs = 0;

  function openTeleprompterIdx(idx) {
    const text = prompterStore[idx] || '';
    openTeleprompterText(text);
  }

  function openTeleprompterText(text) {
    stopTeleprompterScroll();
    const modal = document.getElementById('teleprompterModal');
    const txtEl = document.getElementById('tpText');
    const stage = document.getElementById('tpStage');
    txtEl.textContent = text;
    txtEl.style.fontSize = tpFontSize + 'px';
    stage.scrollTop = 0;
    modal.classList.add('open');
    haptic('success');
  }

  function closeTeleprompter() {
    stopTeleprompterScroll();
    document.getElementById('teleprompterModal').classList.remove('open');
  }

  function resetTeleprompter() {
    stopTeleprompterScroll();
    document.getElementById('tpStage').scrollTop = 0;
  }

  function toggleTpMirror() {
    tpMirror = !tpMirror;
    document.getElementById('tpStage').classList.toggle('mirror', tpMirror);
    document.getElementById('tpMirrorBtn').style.borderColor = tpMirror ? 'var(--accent)' : 'var(--border)';
  }

  function onTpSpeedChange(val) {
    tpSpeed = parseFloat(val) || 1.2;
    document.getElementById('tpSpeedVal').textContent = tpSpeed.toFixed(1) + 'x';
  }

  function changeTpFont(delta) {
    tpFontSize = Math.min(46, Math.max(18, tpFontSize + delta));
    document.getElementById('tpText').style.fontSize = tpFontSize + 'px';
  }

  function startTpWithCountdown() {
    if (tpPlaying) {
      stopTeleprompterScroll();
      return;
    }
    const cd = document.getElementById('tpCountdown');
    let count = 3;
    cd.textContent = count;
    cd.style.display = 'flex';
    const iv = setInterval(() => {
      count--;
      if (count > 0) {
        cd.textContent = count;
        haptic('warning');
      } else {
        clearInterval(iv);
        cd.style.display = 'none';
        startTeleprompterScroll();
      }
    }, 850);
  }

  function toggleTeleprompterPlay() {
    if (document.activeElement === document.getElementById('tpText')) return;
    if (tpPlaying) stopTeleprompterScroll();
    else startTeleprompterScroll();
  }

  function startTeleprompterScroll() {
    tpPlaying = true;
    tpLastTs = performance.now();
    const btn = document.getElementById('tpPlayBtn');
    btn.textContent = '⏸ ПАУЗА (ТАП ПО ЭКРАНУ)';
    btn.classList.add('ghost');

    const stage = document.getElementById('tpStage');
    function step(now) {
      if (!tpPlaying) return;
      const dt = (now - tpLastTs) / 1000;
      tpLastTs = now;
      stage.scrollTop += (38 * tpSpeed) * dt;
      if (stage.scrollTop + stage.clientHeight >= stage.scrollHeight - 4) {
        stopTeleprompterScroll();
        return;
      }
      tpRaf = requestAnimationFrame(step);
    }
    tpRaf = requestAnimationFrame(step);
  }

  function stopTeleprompterScroll() {
    tpPlaying = false;
    if (tpRaf) cancelAnimationFrame(tpRaf);
    const btn = document.getElementById('tpPlayBtn');
    if (btn) {
      btn.textContent = '▶ ПРОДОЛЖИТЬ ЗАПИСЬ';
      btn.classList.remove('ghost');
    }
  }

  // ================= ERID & Watermark Studio =================
  function getEridFooterText() {
    const org = (document.getElementById('eridOrgInp')?.value || '').trim();
    const tok = (document.getElementById('eridTokenInp')?.value || '').trim();
    const link = (document.getElementById('eridLinkInp')?.value || '').trim();
    let lines = [];
    if (link) lines.push(`👉 Ссылка / Артикул: ${link}`);
    if (org || tok) {
      const cleanTok = tok.replace(/^erid:\s*/i, '');
      lines.push(`Реклама. ${org || 'Рекламодатель'}${cleanTok ? `, erid: ${cleanTok}` : ''}`);
    }
    return lines.join('\n');
  }

  function updateEridPreview() {
    const footer = getEridFooterText();
    const box = document.getElementById('eridLiveBox');
    const wmErid = document.getElementById('wmEridInp');
    const tok = (document.getElementById('eridTokenInp')?.value || '').trim().replace(/^erid:\s*/i, '');
    if (wmErid && tok) wmErid.value = `Реклама • erid: ${tok}`;

    if (footer) {
      box.style.display = 'block';
      box.textContent = footer;
    } else {
      box.style.display = 'none';
    }
    drawWatermark();
  }

  let wmPos = 'br';
  let wmBaseImg = null;

  function setWmPos(pos, btn) {
    wmPos = pos;
    document.querySelectorAll('#wmPosChips .chip').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    drawWatermark();
  }

  function loadWatermarkProductImage(url) {
    if (!url) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => { wmBaseImg = img; drawWatermark(); };
    img.src = '/api/proxy-image?url=' + encodeURIComponent(url);
  }

  function uploadCustomWmImage(ev) {
    const file = ev.target.files && ev.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      const img = new Image();
      img.onload = () => { wmBaseImg = img; drawWatermark(); };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function drawWatermark() {
    const canvas = document.getElementById('wmCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = '#0c0c12';
    ctx.fillRect(0, 0, W, H);

    if (wmBaseImg) {
      const ratio = Math.max(W / wmBaseImg.width, H / wmBaseImg.height);
      const nw = wmBaseImg.width * ratio;
      const nh = wmBaseImg.height * ratio;
      ctx.drawImage(wmBaseImg, (W - nw) / 2, (H - nh) / 2, nw, nh);
    } else {
      ctx.strokeStyle = 'rgba(212,255,58,0.18)';
      ctx.lineWidth = 2;
      ctx.strokeRect(24, 24, W - 48, H - 48);
      ctx.fillStyle = '#666677';
      ctx.font = '600 18px Manrope, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Фото товара подгрузится по ссылке', W / 2, H / 2 - 10);
      ctx.fillText('или нажми «📷 Своё фото» ниже', W / 2, H / 2 + 20);
    }

    const wmText = (document.getElementById('wmTextInp')?.value || '').trim();
    const eridText = (document.getElementById('wmEridInp')?.value || '').trim();

    if (wmText) {
      ctx.save();
      if (wmPos === 'center') {
        ctx.translate(W / 2, H / 2);
        ctx.rotate(-Math.PI / 6);
        ctx.font = '800 38px Unbounded, sans-serif';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.28)';
        ctx.textAlign = 'center';
        ctx.fillText(wmText, 0, 0);
      } else {
        ctx.font = '700 20px Manrope, sans-serif';
        const padX = 16, padY = 10;
        const tw = ctx.measureText(wmText).width;
        const bw = tw + padX * 2, bh = 38;
        let bx = W - bw - 20, by = H - bh - 20;
        if (wmPos === 'bl') bx = 20;
        if (wmPos === 'tl') { bx = 20; by = 20; }

        ctx.fillStyle = 'rgba(7, 7, 10, 0.72)';
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, 10);
        ctx.fill();

        ctx.fillStyle = '#d4ff3a';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(wmText, bx + padX, by + bh / 2);
      }
      ctx.restore();
    }

    if (eridText) {
      ctx.save();
      ctx.font = '600 14px JetBrains Mono, monospace';
      const tw = ctx.measureText(eridText).width;
      const bw = tw + 20, bh = 28;
      const bx = 20, by = H - bh - (wmPos === 'bl' ? 68 : 20);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 7);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(eridText, bx + 10, by + bh / 2);
      ctx.restore();
    }
  }

  function downloadWatermarkedImage() {
    const canvas = document.getElementById('wmCanvas');
    const url = canvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sniper_ugc_post.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    showToast('Фото сохранено!');
    haptic('success');
  }

  // ================= Modal & Real Telegram Stars Payment =================
  function openModal(html) {
    document.getElementById('sheet').innerHTML = html;
    document.getElementById('overlay').classList.add('open');
  }
  function closeModal() {
    document.getElementById('overlay').classList.remove('open');
  }
  function updateBalanceUI() {
    const devTop = document.getElementById('devTopBadge');
    const normSub = document.getElementById('normalStudioSub');
    const devCard = document.getElementById('devConsoleCard');
    if (devTop) devTop.classList.toggle('active', !!S.devMode);
    if (normSub) normSub.style.display = S.devMode ? 'none' : 'inline-block';
    if (devCard) devCard.classList.toggle('active', !!S.devMode);

    document.getElementById('balChip').textContent = S.devMode ? '∞ DEV' : S.balance;
    const cb = document.getElementById('cabBalance');
    if (cb) cb.textContent = S.devMode ? '∞' : S.balance;
  }
  function addTx(type, amount) {
    S.tx.unshift({ type: type, amount: amount, ts: Date.now() });
    S.tx = S.tx.slice(0, 40);
  }
  function scrollToTopup() {
    const el = document.getElementById('topupSec');
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function openPay(i) {
    const p = PACKS[i];
    if (!p) return;
    openModal(`
      <h3>${esc(t('pay_title'))}</h3>
      <div class="kv"><span>${esc(t('pay_pack'))}</span><strong>+${p.credits} ${esc(t('pack_credits'))}</strong></div>
      <div class="kv"><span>${esc(t('pay_price'))}</span><strong style="color:var(--accent);">${p.stars} ★ Stars</strong></div>
      <p>${esc(t('pay_note'))}</p>
      <button class="btn" id="payStarsBtn" onclick="payWithTelegramStars(${i})">${esc(t('pay_stars_btn', { s: p.stars }))}</button>
      <button class="btn ghost" onclick="doDemoPay(${i})">${esc(t('pay_demo_btn'))}</button>
      <button class="btn ghost" onclick="closeModal()">${esc(t('cancel'))}</button>
    `);
  }

  async function payWithTelegramStars(i) {
    const p = PACKS[i];
    if (!p) return;
    const btn = document.getElementById('payStarsBtn');
    if (btn) { btn.disabled = true; btn.textContent = t('processing'); }

    try {
      const resp = await fetch('/api/create-stars-invoice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pack_id: p.id,
          user_id: (tgUser && tgUser.id) ? tgUser.id : 0,
          lang: S.lang
        })
      });
      const data = await resp.json();
      if (!resp.ok || !data.invoice_link) {
        throw new Error(data.detail || 'Could not create Stars invoice');
      }

      if (tg && tg.openInvoice) {
        tg.openInvoice(data.invoice_link, function(status) {
          if (status === 'paid') {
            creditUserBalance(p.credits);
          } else if (btn) {
            btn.disabled = false;
            btn.textContent = t('pay_stars_btn', { s: p.stars });
          }
        });
      } else {
        window.open(data.invoice_link, '_blank');
        if (btn) {
          btn.disabled = false;
          btn.textContent = t('pay_stars_btn', { s: p.stars });
        }
      }
    } catch (e) {
      showToast(e.message || 'Ошибка создания инвойса');
      if (btn) {
        btn.disabled = false;
        btn.textContent = t('pay_stars_btn', { s: p.stars });
      }
    }
  }

  function doDemoPay(i) {
    const p = PACKS[i];
    if (!p) return;
    creditUserBalance(p.credits);
  }

  function creditUserBalance(credits) {
    S.balance += credits;
    addTx('topup', credits);
    saveState();
    updateBalanceUI();
    renderCabinet();
    closeModal();
    showToast(t('pay_done', { n: credits }));
    haptic('success');
  }

  function openNoFunds(needed) {
    openModal(`
      <h3>${esc(t('nofunds_title'))}</h3>
      <p>${esc(t('nofunds_text', { n: needed || COST }))}</p>
      <button class="btn" onclick="closeModal(); showView('cabinet'); setTimeout(scrollToTopup, 80);">${esc(t('topup'))}</button>
      <button class="btn ghost" onclick="closeModal()">${esc(t('cancel'))}</button>
    `);
  }

  function applyPromo() {
    const inp = document.getElementById('promoInp');
    const code = (inp.value || '').trim().toUpperCase();
    if (!code) return;

    // Secret promo code to toggle Developer Mode
    if (code === 'DEVMODE' || code === 'DEVELOPER') {
      inp.value = '';
      toggleDevMode(true);
      renderCabinet();
      return;
    }

    if (S.promoUsed) { showToast(t('promo_used')); haptic('error'); return; }
    const bonus = PROMO_CODES[code];
    if (!bonus) { showToast(t('promo_bad')); haptic('error'); return; }
    S.promoUsed = true;
    S.balance += bonus;
    addTx('promo', bonus);
    saveState();
    updateBalanceUI();
    inp.value = '';
    renderCabinet();
    showToast(t('promo_ok', { n: bonus }));
    haptic('success');
  }

  // ================= Cabinet Render =================
  function renderCabinet() {
    const name = tgUser ? ([tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ') || tgUser.username || t('guest')) : t('guest');
    const sub = tgUser ? (tgUser.username ? '@' + tgUser.username : 'ID ' + tgUser.id) : 'Telegram Mini App';
    const initial = (name || '?').trim().charAt(0).toUpperCase() || '?';
    const avatar = (tgUser && tgUser.photo_url)
      ? `<div class="avatar"><img src="${esc(tgUser.photo_url)}" alt="" onerror="this.parentNode.textContent='${esc(initial)}'"></div>`
      : `<div class="avatar">${esc(initial)}</div>`;
    const roleTag = S.devMode
      ? `<span class="tag" style="background:var(--accent); color:#000; font-weight:800; border:none;">⚡ OWNER // DEV MODE</span>`
      : `<span class="tag">${esc(t('plan'))}</span>`;

    document.getElementById('cabProfile').innerHTML = `
      ${avatar}
      <div style="min-width:0;">
        <div style="font-weight:800; font-size:16px;">${esc(name)}</div>
        <div style="color:var(--muted); font-size:12px; margin-top:2px;">${esc(sub)} · ${roleTag}</div>
      </div>`;

    updateBalanceUI();

    document.getElementById('cabStats').innerHTML = `
      <div class="stat"><div class="v">${S.gens}</div><div class="k">${esc(t('gens'))}</div></div>
      <div class="stat"><div class="v">${S.spent}</div><div class="k">${esc(t('spent'))}</div></div>`;

    document.getElementById('cabPacks').innerHTML = PACKS.map((p, i) => `
      <button class="pack ${p.pop ? 'pop' : ''}" onclick="openPay(${i})">
        ${p.pop ? `<span class="badge">${esc(t('popular'))}</span>` : ''}
        <div class="pc">+${p.credits}</div>
        <div class="pk">${esc(t('pack_credits'))}</div>
        <div class="pp">★ ${p.stars} Stars</div>
      </button>`).join('');

    const hist = document.getElementById('cabHistory');
    if (!S.history.length) {
      hist.innerHTML = `<div class="empty">${esc(t('history_empty'))}</div>`;
    } else {
      hist.innerHTML = S.history.map((it, i) => `
        <div class="list-item">
          <div class="li-main">
            <div class="li-title">${esc(it.title)}</div>
            <div class="li-sub">${esc(fmtDate(it.ts))} · ${esc(it.query)}</div>
          </div>
          <div style="display:flex; gap:6px; flex-shrink:0;">
            <button class="copy-btn" onclick="openHistory(${i})">${esc(t('open'))}</button>
            <button class="copy-btn" onclick="delHistory(${i})">✕</button>
          </div>
        </div>`).join('');
    }

    const tx = document.getElementById('cabTx');
    if (!S.tx.length) {
      tx.innerHTML = `<div class="empty">${esc(t('tx_empty'))}</div>`;
    } else {
      tx.innerHTML = S.tx.slice(0, 15).map(x => `
        <div class="list-item">
          <div class="li-main">
            <div class="li-title">${esc(t('tx_' + x.type))}</div>
            <div class="li-sub">${esc(fmtDate(x.ts))}</div>
          </div>
          <div class="${x.amount >= 0 ? 'plus' : 'minus'}">${x.amount >= 0 ? '+' : ''}${x.amount}</div>
        </div>`).join('');
    }
  }
  function openHistory(i) {
    const it = S.history[i];
    if (!it) return;
    lastResult = it.data;
    renderResult(it.data);
    showView('studio');
    const res = document.getElementById('results');
    res.style.display = 'block';
    setTimeout(() => res.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
  }
  function delHistory(i) {
    S.history.splice(i, 1);
    saveState();
    renderCabinet();
  }
  function clearHistory() {
    S.history = [];
    saveState();
    renderCabinet();
    showToast(t('cleared'));
  }

  // ================= Generation Pipeline =================
  function showErr(msg) {
    const b = document.getElementById('errBox');
    b.textContent = msg;
    b.style.display = 'block';
    haptic('error');
  }
  function hideErr() { document.getElementById('errBox').style.display = 'none'; }

  async function postJsonWithRetry(url, body, signal) {
    const doFetch = () => fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: signal
    });
    try {
      return await doFetch();
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      await new Promise(r => setTimeout(r, 3500));
      return await doFetch();
    }
  }

  async function startGeneration() {
    hideErr();
    const q = document.getElementById('queryInp').value.trim();
    if (!q) { showErr(t('err_query')); return; }
    if (!S.devMode && S.balance < COST) { openNoFunds(COST); return; }

    const btn = document.getElementById('submitBtn');
    const res = document.getElementById('results');

    btn.disabled = true;
    res.style.display = 'none';
    startAiTerminal(q);

    const ctrl = new AbortController();
    const abortTimer = setTimeout(() => ctrl.abort(), 120000);

    try {
      const resp = await postJsonWithRetry('/api/analyze', { query: q, format_type: curFmt, tone: curTone, lang: S.lang }, ctrl.signal);
      let payload = {};
      try { payload = await resp.json(); } catch (e) {}
      if (!resp.ok) throw new Error(payload.detail || t('err_generic'));

      if (!S.devMode) {
        S.balance -= COST;
        S.spent += COST;
        addTx('gen', -COST);
      }
      S.gens += 1;
      S.history.unshift({
        id: Date.now(),
        ts: Date.now(),
        title: payload.product_title || q.slice(0, 40),
        query: q.slice(0, 60),
        data: payload
      });
      S.history = S.history.slice(0, 15);
      saveState();
      updateBalanceUI();

      lastResult = payload;
      renderResult(payload);
      res.style.display = 'block';
      if (payload.meta_image) loadWatermarkProductImage(payload.meta_image);
      haptic('success');
    } catch (e) {
      if (e && e.name === 'AbortError') showErr(t('err_timeout'));
      else showErr((e && e.message) || t('err_generic'));
    } finally {
      clearTimeout(abortTimer);
      stopAiTerminal();
      btn.disabled = false;
    }
  }

  async function generateStandaloneTgPosts() {
    const q = document.getElementById('tgQueryInp').value.trim();
    if (!q) { showToast(t('err_query')); haptic('error'); return; }
    if (!S.devMode && S.balance < TG_COST) { openNoFunds(TG_COST); return; }

    const btn = document.getElementById('tgGenBtn');
    const out = document.getElementById('tgPostsOutput');
    btn.disabled = true;
    out.innerHTML = `<div class="panel" style="text-align:center; color:var(--accent); font-family:var(--mono);">⚡ ПИШУ 3 РЕКЛАМНЫХ ПОСТА С МАРКИРОВКОЙ...</div>`;

    try {
      const resp = await postJsonWithRetry('/api/tg-post', { query: q, style: curTone, lang: S.lang });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || t('err_generic'));

      if (!S.devMode) {
        S.balance -= TG_COST;
        S.spent += TG_COST;
        addTx('gen', -TG_COST);
      }
      S.gens += 1;
      saveState();
      updateBalanceUI();

      if (data.meta_image) loadWatermarkProductImage(data.meta_image);
      out.innerHTML = renderTgPostsCards(data.tg_posts || []);
      haptic('success');
    } catch (e) {
      out.innerHTML = `<div class="err-box" style="display:block;">${esc(e.message || t('err_generic'))}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  function renderTgPostsCards(posts) {
    const footer = getEridFooterText();
    return posts.map((p, idx) => {
      const fullPost = `${p.headline}\n\n${p.body}\n\n${p.cta}${footer ? '\n\n' + footer : ''}`;
      const tpIdx = regTp(`${p.headline}.\n\n${p.body}\n\n${p.cta}`);
      const cpIdx = reg(fullPost);
      return `
        <div class="card">
          <div class="card-top">
            <div>
              <span class="tag" style="color:var(--accent); border-color:var(--border-strong);">ПОСТ #0${idx + 1} · ${esc(p.style)}</span>
            </div>
            <div class="action-row">
              <button class="prompter-btn" onclick="openTeleprompterIdx(${tpIdx})">${esc(t('prompter'))}</button>
              <button class="copy-btn" onclick="copyIdx(${cpIdx})">${esc(t('copy'))}</button>
            </div>
          </div>
          <div style="font-weight:800; font-size:14.5px; margin-bottom:6px;">${esc(p.headline)}</div>
          <div class="tg-post-body">${esc(p.body)}</div>
          <div style="font-weight:700; font-size:13px; color:var(--accent);">${esc(p.cta)}</div>
          ${footer ? `<div class="erid-preview" style="display:block;">${esc(footer)}</div>` : ''}
        </div>
      `;
    }).join('');
  }

  // ================= Render Studio Result =================
  function renderResult(d) {
    copyStore.length = 0;
    prompterStore.length = 0;

    const imgHtml = d.meta_image ? `<img class="prod-thumb" src="${esc(d.meta_image)}" alt="" onerror="this.style.display='none'">` : '';
    const price = d.meta_price ? `<span class="tag" style="color:var(--ok);">${esc(d.meta_price)}</span>` : '';
    document.getElementById('pHeader').innerHTML = `
      ${imgHtml}
      <div style="min-width:0;">
        <div style="font-weight:800; font-size:14.5px; margin-bottom:6px;">${esc(d.product_title)}</div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
          <span class="tag">${esc(d.meta_source || 'WEB')}</span>
          <span class="tag">${esc(d.category || '')}</span>
          ${price}
        </div>
      </div>`;

    // 1. Scripts + Teleprompter Button
    document.getElementById('tab-scripts').innerHTML = (d.scripts || []).map((sc, i) => {
      const full = `${t('script')}: ${sc.title}\n${t('hook')}: ${sc.hook}\n` +
        (sc.scenes || []).map(s => `[${s.timecode}] ${s.visual} | ${t('voice')}: ${s.voiceover}`).join('\n') +
        `\nCTA: ${sc.cta}`;

      const teleprompterVoiceText = [
        sc.hook,
        ...(sc.scenes || []).map(s => s.voiceover),
        sc.cta
      ].filter(Boolean).join('\n\n');

      const tpIdx = regTp(teleprompterVoiceText);
      const cpAllIdx = reg(full);

      const scenes = (sc.scenes || []).map(s => `
        <div class="scene-item">
          <div class="mono" style="color:var(--accent); font-size:10.5px; margin-bottom:4px;">⏱ ${esc(s.timecode)}</div>
          <div><strong>${esc(t('frame'))}:</strong> ${esc(s.visual)}</div>
          <div style="margin:4px 0; color:#fff;"><strong>${esc(t('voice'))}:</strong> «${esc(s.voiceover)}»</div>
          <div><strong>${esc(t('screen_text'))}:</strong> ${esc(s.on_screen_text)}</div>
          <div class="prompt-box mono">
            <span>🎥 AI: ${esc(s.ai_video_prompt)}</span>
            <button class="copy-btn" onclick="copyIdx(${reg(s.ai_video_prompt)})">${esc(t('copy'))}</button>
          </div>
        </div>`).join('');

      return `
        <div class="card">
          <div class="card-top">
            <div style="min-width:0;">
              <div style="font-weight:800; text-transform:uppercase;">0${i + 1} // ${esc(sc.title)}</div>
              <div class="mono" style="font-size:10px; color:var(--muted);">${esc(sc.angle)} · <span style="color:var(--accent);">${esc(sc.virality_score)}% ${esc(t('virality'))}</span></div>
            </div>
            <div class="action-row">
              <button class="prompter-btn" onclick="openTeleprompterIdx(${tpIdx})">${esc(t('prompter'))}</button>
              <button class="copy-btn" onclick="copyIdx(${cpAllIdx})">${esc(t('copy_all'))}</button>
            </div>
          </div>
          <div class="hook-box"><strong>${esc(t('hook'))}:</strong> «${esc(sc.hook)}»</div>
          ${scenes}
          <div style="font-size:12.5px; margin-top:8px;"><strong>CTA:</strong> ${esc(sc.cta)}</div>
        </div>`;
    }).join('');

    // 2. TG Ad Posts inside Studio
    const tgPosts = d.tg_posts || [];
    document.getElementById('tab-tgposts').innerHTML = tgPosts.length
      ? renderTgPostsCards(tgPosts)
      : `<div class="card"><div class="empty">Перейди во вкладку «TG Посты & ERID» снизу для генерации постов с маркировкой.</div></div>`;

    // 3. Hooks
    document.getElementById('tab-hooks').innerHTML = `
      <div class="card">
        ${(d.viral_hooks || []).map((hk, i) => {
          const hTpIdx = regTp(hk);
          return `
          <div class="list-item">
            <div style="font-size:13px;"><strong>#0${i + 1}</strong> «${esc(hk)}»</div>
            <div class="action-row">
              <button class="prompter-btn" onclick="openTeleprompterIdx(${hTpIdx})">🎬</button>
              <button class="copy-btn" onclick="copyIdx(${reg(hk)})">${esc(t('copy'))}</button>
            </div>
          </div>`;
        }).join('')}
      </div>`;

    // 4. AI video prompts
    document.getElementById('tab-prompts').innerHTML = `
      <div class="card">
        ${(d.video_generator_prompts || []).map(p => `
          <div class="prompt-box mono" style="margin-bottom:8px;">
            <span>${esc(p)}</span>
            <button class="copy-btn" onclick="copyIdx(${reg(p)})">${esc(t('copy'))}</button>
          </div>`).join('')}
      </div>`;

    // 5. Audience & SEO
    const ht = (d.hashtags || []).join(' ');
    const seoFull = `${d.seo_caption || ''}\n\n${ht}`;
    const pains = (d.pain_points || []).map(x => `<li>${esc(x)}</li>`).join('');
    const feats = (d.killer_features || []).map(x => `<li>${esc(x)}</li>`).join('');
    document.getElementById('tab-seo').innerHTML = `
      <div class="card">
        <div class="card-top"><strong>${esc(t('audience'))}</strong></div>
        <div style="font-size:13px;">${esc(d.target_audience)}</div>
      </div>
      ${pains ? `<div class="card"><div class="card-top"><strong>${esc(t('pains'))}</strong></div><ul style="font-size:13px; padding-left:18px;">${pains}</ul></div>` : ''}
      ${feats ? `<div class="card"><div class="card-top"><strong>${esc(t('features'))}</strong></div><ul style="font-size:13px; padding-left:18px;">${feats}</ul></div>` : ''}
      <div class="card">
        <div class="card-top">
          <strong>${esc(t('seo_title'))}</strong>
          <button class="copy-btn" onclick="copyIdx(${reg(seoFull)})">${esc(t('copy_seo'))}</button>
        </div>
        <div style="font-size:13px; margin-bottom:8px; white-space:pre-line;">${esc(d.seo_caption)}</div>
        <div class="mono" style="font-size:11px; color:var(--muted);">${esc(ht)}</div>
      </div>`;
  }

  // ================= Init =================
  updateBalanceUI();
  applyLang();
  drawWatermark();
  fetch('/health').catch(() => {});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    html_out = FRONTEND_UI.replace("__ADMIN_TG_ID__", ADMIN_TG_ID)
    return HTMLResponse(
        content=html_out,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
