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
# SETTINGS & MODELS
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
APP_URL = os.getenv("RENDER_EXTERNAL_URL", "https://sniper-ugc.onrender.com").strip()
APP_URL_VERSIONED = APP_URL.rstrip("/") + "/?v=15"

ADMIN_TG_ID = os.getenv("ADMIN_TG_ID", "").strip()
SUPPORT_USERNAME = "onadash"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.0-flash,gemini-2.5-flash-lite,gemini-2.0-flash-lite"
    ).split(",")
    if m.strip()
]

REQUEST_TIMEOUT_MS = 42000
TOTAL_BUDGET_SEC = 90
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
    except Exception:
        return genai.Client(api_key=GEMINI_API_KEY)


ai_client = make_ai_client()

_MODEL_CACHE = {"ts": 0.0, "models": []}
_BAD_WORDS = (
    "image", "tts", "audio", "live", "embed", "vision", "robotics",
    "computer", "learnlm", "gemma", "aqa", "imagen", "veo", "exp", "thinking"
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
    found.sort(key=lambda n: ("preview" in n, "lite" in n, -_model_version(n)))
    return found


async def get_model_candidates() -> list:
    static = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    loop = asyncio.get_running_loop()
    now = loop.time()
    if not _MODEL_CACHE["models"] or now - _MODEL_CACHE["ts"] > 3600:
        try:
            models = await asyncio.wait_for(asyncio.to_thread(discover_models_sync), timeout=8)
            _MODEL_CACHE["models"] = models
        except Exception:
            pass
        _MODEL_CACHE["ts"] = now
    discovered = _MODEL_CACHE["models"]
    combined = static + [m for m in discovered if m not in static]
    return list(dict.fromkeys(combined))[:5]


# =====================================================================
# SCHEMAS & PACKAGES
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
    action: str = "shorter"
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
        "desc_ru": "Пополнение баланса на 100 ★ (1 полный сценарий с суфлёром или TG-пост)",
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
        "title_ru": "1000 ★ SNIPER PRO (-15%)",
        "title_en": "1000 ★ SNIPER PRO (-15%)",
        "desc_ru": "PRO-пакет для селлеров и креаторов: 1000 ★ на баланс",
        "desc_en": "PRO pack for sellers and creators: 1000 ★ balance",
    },
}


class AppError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


ERR = {
    "ru": {
        "no_key": "GEMINI_API_KEY не задан на сервере (Render → Environment).",
        "empty": "Нейросеть вернула пустой ответ. Попробуй ещё раз — звёзды не списались.",
        "bad_json": "Ошибка форматирования ответа ИИ. Попробуй ещё раз.",
        "busy": "Высокая нагрузка на облако ИИ. Подожди 3 секунды и повтори запрос.",
        "quota": "Достигнут минутный лимит запросов к API. Попробуй через 15 секунд.",
        "key": "Проблема с ключом GEMINI_API_KEY в настройках сервера.",
        "no_model": "Модель недоступна. Проверь настройки Environment.",
        "timeout": "Сервер долго отвечал. Попробуй ещё раз.",
        "short_query": "Вставь ссылку на товар, артикул или описание.",
        "unknown": "Ошибка обработки",
    },
    "en": {
        "no_key": "GEMINI_API_KEY is not set on the server (Render → Environment).",
        "empty": "Model returned an empty response. Please try again.",
        "bad_json": "Invalid response format from AI. Please try again.",
        "busy": "AI cloud is busy right now. Wait 3 seconds and retry.",
        "quota": "API rate limit reached. Try again in 15 seconds.",
        "key": "Invalid GEMINI_API_KEY configuration.",
        "no_model": "No available Gemini model found.",
        "timeout": "Request timed out. Please try again.",
        "short_query": "Enter a product link, SKU or description.",
        "unknown": "Processing error",
    },
}

TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "high demand", "overloaded", "429",
    "RESOURCE_EXHAUSTED", "500", "INTERNAL", "DEADLINE", "timed out",
    "timeout", "Timeout", "ReadError", "ConnectError", "RemoteProtocolError",
)
GONE_MARKERS = ("NOT_FOUND", "404", "no longer available", "not supported")


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


def error_text(e: Exception, lang: str = "ru") -> str:
    lang = "en" if lang == "en" else "ru"
    code = classify(e)
    text = ERR[lang].get(code, ERR[lang]["unknown"])
    if code == "unknown":
        text += f": {str(e)[:140]}"
    return text


def norm_lang(code: str) -> str:
    return "ru" if (code or "")[:2].lower() in ("ru", "uk", "be", "kk") else "en"


# =====================================================================
# PRODUCT PARSER (WB / OZON / WEB) - CLEAN OF RECOMMENDATION NOISE
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
    except Exception:
        pass

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
            info["description"] = (card.get("description") or "")[:1200]
            opts = card.get("options") or []
            info["features"] = [
                f"{o.get('name')}: {o.get('value')}"
                for o in opts[:10]
                if o.get("name") and o.get("value")
            ]
        except Exception:
            pass

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
    except Exception:
        pass

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
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=7.5) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    og_title = soup.find("meta", property="og:title")
                    og_desc = soup.find("meta", property="og:description")
                    og_img = soup.find("meta", property="og:image")
                    meta_desc = soup.find("meta", attrs={"name": "description"})

                    if og_title and og_title.get("content"):
                        fallback_data["title"] = og_title["content"].strip()
                    elif soup.title and soup.title.string:
                        fallback_data["title"] = soup.title.string.strip()

                    desc_val = ""
                    if og_desc and og_desc.get("content"):
                        desc_val = og_desc["content"].strip()
                    elif meta_desc and meta_desc.get("content"):
                        desc_val = meta_desc["content"].strip()
                    fallback_data["description"] = desc_val[:900]

                    if og_img and og_img.get("content"):
                        fallback_data["image"] = og_img["content"].strip()

                    for sc_tag in soup.find_all("script", type="application/ld+json"):
                        try:
                            ld = json.loads(sc_tag.string or "{}")
                            if isinstance(ld, dict) and ld.get("@type") in ("Product", "ProductGroup"):
                                fallback_data["title"] = ld.get("name") or fallback_data["title"]
                                fallback_data["description"] = (ld.get("description") or fallback_data["description"])[:1000]
                                brand_obj = ld.get("brand")
                                if isinstance(brand_obj, dict):
                                    fallback_data["brand"] = brand_obj.get("name", "")
                                elif isinstance(brand_obj, str):
                                    fallback_data["brand"] = brand_obj
                        except Exception:
                            pass
        except Exception:
            pass

        _PREVIEW_CACHE[cache_key] = fallback_data
        if len(_PREVIEW_CACHE) > 80:
            _PREVIEW_CACHE.popitem(last=False)
        return fallback_data

    return {
        "source": "Text",
        "id": "",
        "title": raw[:80],
        "brand": "",
        "category": "",
        "price": "",
        "description": raw[:1500],
        "features": [],
        "image": ""
    }


# =====================================================================
# GEMINI GENERATION (STRICT LANGUAGE + ZERO HALLUCINATION)
# =====================================================================
SYSTEM_INSTRUCTION = """
You are an elite creative producer of viral short-form videos (TikTok, Reels, Shorts) and Telegram Ad integrations.

STRICT RULES:
1. LANGUAGE: Write ALL text fields strictly in the requested OUTPUT LANGUAGE (if Russian is requested, write 100% in natural modern Russian; if English is requested, write in English). Only `ai_video_prompt` and `video_generator_prompts` are always in English.
2. ZERO HALLUCINATIONS: Describe ONLY the exact physical product specified in `title`, `category`, `description`, and `features`. NEVER invent unrelated specs (e.g., never mention USB-C chargers, cables, or batteries if the product is a home fragrance diffuser, clothing, or cosmetics!).
3. OUTPUT FORMAT: Return STRICTLY valid JSON without markdown fences.

JSON SCHEMA:
{
  "product_title": "string",
  "category": "string",
  "target_audience": "string",
  "pain_points": ["string", "string", "string"],
  "killer_features": ["string", "string", "string"],
  "viral_hooks": ["string", "string", "string", "string", "string"],
  "scripts": [
    {
      "title": "string",
      "angle": "string",
      "virality_score": 96,
      "sound_vibe": "string",
      "hook": "string",
      "scenes": [
        {
          "timecode": "0:00-0:03",
          "visual": "string",
          "voiceover": "string (natural spoken phrase for teleprompter, max 18 words)",
          "on_screen_text": "string (max 6 words)",
          "ai_video_prompt": "Cinematic b-roll prompt in English for Kling/Runway"
        }
      ],
      "cta": "string"
    }
  ],
  "tg_posts": [
    {
      "style": "string",
      "headline": "string",
      "body": "string (specific to this exact product, with bullet points, no generic fluff)",
      "cta": "string"
    }
  ],
  "video_generator_prompts": ["English prompt 1", "English prompt 2"],
  "seo_caption": "string",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"]
}
"""

REWRITE_SYSTEM_INSTRUCTION = """
You are a top short-form video script doctor.
Rewrite the provided script strictly following the user's instruction while staying 100% faithful to the product.
Write all fields in the requested OUTPUT LANGUAGE (except `ai_video_prompt` which stays in English).
Return STRICTLY valid JSON:
{
  "script": {
    "title": "string",
    "angle": "string",
    "virality_score": 97,
    "sound_vibe": "string",
    "hook": "string",
    "scenes": [
      {
        "timecode": "0:00-0:02",
        "visual": "string",
        "voiceover": "string",
        "on_screen_text": "string",
        "ai_video_prompt": "English prompt"
      }
    ],
    "cta": "string"
  }
}
"""

TG_POST_SYSTEM_INSTRUCTION = """
You are an expert copywriter for Telegram blogger integrations and marketplace seeding.
RULES:
1. Write strictly in the requested OUTPUT LANGUAGE.
2. Write ONLY about the exact provided product and its real attributes. Zero hallucinated features from unrelated electronics!
3. Generate 3 distinct posts: Native personal recommendation, Structured "Find of the Day" with bullets, and Value/Problem-solving offer.
Return STRICTLY valid JSON:
{
  "product_title": "string",
  "tg_posts": [
    {
      "style": "string",
      "headline": "string",
      "body": "string",
      "cta": "string"
    }
  ]
}
"""

FORMAT_LABELS = {
    "ugc": "Нативный честный UGC-обзор от лица реального покупателя",
    "problem_solution": "Острая жизненная боль -> Решение проблемы этим товаром",
    "unboxing": "Эстетичная ASMR-распаковка и первая эмоция",
    "aesthetic": "Визуальный минимализм / Pinterest & TikTok Aesthetic",
    "storytelling": "Короткий сторителлинг с интригой в начале и развязкой",
}
TONE_LABELS = {
    "native": "Искренний, дружеский, разговорный ('как голосовое подруге')",
    "aggressive": "Дерзкий, триггерный, провокационный, максимальное удержание",
    "humor": "Жизненная ирония, легкий юмор и мемная подача",
    "luxury": "Премиальный, спокойный, дорогой минимализм",
}
REWRITE_ACTIONS = {
    "shorter": "Make the script 40% shorter, faster-paced, cut all fluff, leave only punchy phrases.",
    "longer": "Make the script more detailed: expand product usage details, strong arguments, and 4-5 rich scenes.",
    "bolder": "Make the hook and voiceover much bolder, provocative, and clickbait to stop the scroll immediately.",
    "native": "Rewrite the voiceover to sound 100% natural and casual like a voice message to a close friend, zero ad cliches.",
    "funny": "Add relatable everyday humor, self-irony, and charismatic delivery to the hook and voiceover.",
}


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
        temperature=0.72,
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
        for attempt in range(2):
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
                continue
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in GONE_MARKERS):
                    gone_error = e
                    break
                if any(k in msg for k in TRANSIENT_MARKERS):
                    busy_error = e
                    await asyncio.sleep(0.7)
                    break
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
    out_lang = "ENGLISH" if lang == "en" else "RUSSIAN (СТРОГО НА РУССКОМ ЯЗЫКЕ)"
    clean_context = {
        "source": data.get("source", ""),
        "sku": data.get("id", ""),
        "title": data.get("title", ""),
        "brand": data.get("brand", ""),
        "category": data.get("category", ""),
        "price": data.get("price", ""),
        "description": data.get("description", ""),
        "features": data.get("features", []),
    }
    user_prompt = f"""
EXACT PRODUCT DATA:
{json.dumps(clean_context, ensure_ascii=False, indent=2)}

SELECTED VIDEO FORMAT: {FORMAT_LABELS.get(format_type, format_type)}
SELECTED TONE: {TONE_LABELS.get(tone, tone)}
OUTPUT LANGUAGE: {out_lang}

TASK:
1. Write all text fields strictly in {out_lang}.
2. Use ONLY real attributes of this exact product from the data above.
3. Generate 2 detailed scripts (4 scenes each), 5 viral hooks, 3 audience pains, 3 real product features, and 3 ready-to-publish Telegram ad posts (tg_posts).
"""
    parsed = await _call_gemini_json(SYSTEM_INSTRUCTION, user_prompt, require_key="scripts")
    parsed["meta_source"] = data.get("source", "Web")
    parsed["meta_image"] = data.get("image", "")
    parsed["meta_price"] = data.get("price", "")
    parsed["meta_brand"] = data.get("brand", "")
    parsed["meta_id"] = data.get("id", "")
    return parsed


async def rewrite_single_script(req: RewriteScriptRequest) -> dict:
    out_lang = "ENGLISH" if req.lang == "en" else "RUSSIAN (СТРОГО НА РУССКОМ ЯЗЫКЕ)"
    instruction = (
        req.custom_instruction.strip()
        if req.action == "custom" and req.custom_instruction.strip()
        else REWRITE_ACTIONS.get(req.action, REWRITE_ACTIONS["shorter"])
    )
    user_prompt = f"""
PRODUCT: {req.product_title}
CURRENT SCRIPT JSON:
{json.dumps(req.script, ensure_ascii=False)}

REWRITE INSTRUCTION:
{instruction}

OUTPUT LANGUAGE: {out_lang} (keep ai_video_prompt in English). Return inside {{"script": ...}}.
"""
    parsed = await _call_gemini_json(REWRITE_SYSTEM_INSTRUCTION, user_prompt, require_key="script")
    return parsed["script"]


async def generate_tg_posts_only(data: dict, style: str, lang: str = "ru") -> dict:
    out_lang = "ENGLISH" if lang == "en" else "RUSSIAN (СТРОГО НА РУССКОМ ЯЗЫКЕ)"
    clean_context = {
        "title": data.get("title", ""),
        "brand": data.get("brand", ""),
        "category": data.get("category", ""),
        "price": data.get("price", ""),
        "description": data.get("description", ""),
        "features": data.get("features", []),
    }
    user_prompt = f"""
EXACT PRODUCT DATA:
{json.dumps(clean_context, ensure_ascii=False, indent=2)}

PREFERRED TONE: {TONE_LABELS.get(style, style)}
OUTPUT LANGUAGE: {out_lang}

Write 3 distinct Telegram ad posts strictly in {out_lang} and strictly about this specific product.
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


async def run_pipeline(query: str, format_type: str, tone: str, lang: str = "ru") -> dict:
    lang = "en" if lang == "en" else "ru"
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
# TELEGRAM BOT HANDLERS
# =====================================================================
WELCOME_MSG_RU = (
    "🏴 <b>SNIPER // TITANIUM STUDIO 6.0</b>\n"
    "<i>Профессиональная AI-платформа создания виральных сценариев, телесуфлёра и рекламных посевов для TikTok, Reels, Shorts и Telegram</i>\n\n"
    "━━━━━━━━━━━━━━━\n"
    "⚡ <b>ЧТО ВНУТРИ СТУДИИ:</b>\n"
    "• <b>Мгновенный парсер WB / Ozon / Сайтов:</b> вставляешь ссылку — карточка товара с фото и ценой подтягивается автоматически.\n"
    "• <b>Виральные сценарии + AI-Рерайт:</b> хуки, раскадровка и кнопки быстрой переработки <i>«Короче / Дерзко / Нативнее / Юмор»</i>.\n"
    "• <b>Выделенный Телесуфлёр:</b> выбирай любой сценарий из истории и читай с экрана при записи видео.\n"
    "• <b>TG-Посты + ERID + Водяной знак:</b> генерация постов под рекламу с маркировкой ОРД и штампом на фото.\n"
    "━━━━━━━━━━━━━━━\n"
    "💎 <b>ТАРИФЫ:</b>\n"
    "🎬 Полный разбор и сценарии — <b>100 ★</b>\n"
    "📢 Пакет из 3 рекламных TG-постов — <b>75 ★</b>\n"
    "🎁 Бонус на старте — <b>100 ★ бесплатно!</b>\n"
    "💬 Поддержка 24/7: @onadash\n\n"
    "👇 <b>Отправь ссылку/артикул прямо сюда или открой студию:</b>"
)


@dp.pre_checkout_query()
async def on_pre_checkout(pre_checkout_query: PreCheckoutQuery):
    await pre_checkout_query.answer(ok=True)


@dp.message(F.successful_payment)
async def on_successful_payment(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Открыть SNIPER STUDIO", web_app=WebAppInfo(url=APP_URL_VERSIONED))]
    ])
    await message.answer(
        "✅ <b>Оплата прошла успешно!</b>\nЗвёзды зачислены на твой баланс в приложении SNIPER UGC.",
        reply_markup=kb,
        parse_mode="HTML"
    )


@dp.message(Command("start"))
async def bot_start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Запустить SNIPER STUDIO", web_app=WebAppInfo(url=APP_URL_VERSIONED))],
        [
            InlineKeyboardButton(text="⭐ Купить 100 ★", callback_data="buy:pack_100"),
            InlineKeyboardButton(text="🔥 Купить 300 ★ (Хит)", callback_data="buy:pack_300"),
        ],
        [
            InlineKeyboardButton(text="💡 Как это работает", callback_data="how_it_works"),
            InlineKeyboardButton(text="💬 Саппорт @onadash", url=f"[https://t.me/](https://t.me/){SUPPORT_USERNAME}"),
        ],
    ])
    await message.answer(WELCOME_MSG_RU, reply_markup=kb, parse_mode="HTML")


@dp.callback_query(F.data == "how_it_works")
async def cb_how_it_works(call: CallbackQuery):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Открыть студию", web_app=WebAppInfo(url=APP_URL_VERSIONED))],
        [InlineKeyboardButton(text="💬 Написать @onadash", url=f"[https://t.me/](https://t.me/){SUPPORT_USERNAME}")],
    ])
    await call.message.answer(
        "🎬 <b>КАК СОЗДАВАТЬ РОЛИКИ И ПОСТЫ ЗА 1 МИНУТУ:</b>\n\n"
        "1️⃣ Скопируй ссылку на товар с Wildberries, Ozon или любого сайта.\n"
        "2️⃣ Вставь её в студию — приложение сразу покажет превью товара.\n"
        "3️⃣ При необходимости открой <b>«⚙️ Параметры сценария»</b> и выбери подачу.\n"
        "4️⃣ Получи готовые сценарии, нажми <b>«✂️ Короче»</b> или <b>«🔥 Дерзко»</b> для мгновенного AI-рерайта и запусти встроенный <b>Суфлёр</b>!",
        reply_markup=kb,
        parse_mode="HTML"
    )
    await call.answer()


@dp.callback_query(F.data.startswith("buy:"))
async def cb_buy_pack(call: CallbackQuery):
    pack_id = call.data.split(":", 1)[1]
    pack = STAR_PACKS.get(pack_id)
    if not pack:
        await call.answer("Пакет не найден", show_alert=True)
        return
    uid = call.from_user.id if call.from_user else 0
    await call.message.answer_invoice(
        title=pack["title_ru"],
        description=pack["desc_ru"],
        payload=f"{pack_id}:{uid}:{pack['credits']}",
        provider_token="",
        currency="XTR",
        prices=[LabeledPrice(label=pack["title_ru"], amount=pack["stars"])],
    )
    await call.answer()


@dp.message()
async def bot_quick_message(message: types.Message):
    txt = (message.text or "").strip()
    if not txt:
        return
    wait_msg = await message.answer("🔍 <b>Сканирую карточку товара и пишу сценарий на русском...</b>", parse_mode="HTML")
    try:
        res = await run_pipeline(txt, "ugc", "native", "ru")
        scripts = res.get("scripts") or []
        if not scripts:
            raise AppError("bad_json")
        s1 = scripts[0]

        hooks = "\n".join(
            f"• <i>«{h(str(x))}»</i>" for x in (res.get("viral_hooks") or [])[:3]
        )
        scenes_txt = "\n".join(
            f"<code>[{h(str(sc.get('timecode', '')))}]</code> {h(str(sc.get('visual', '')))}\n🗣 <i>«{h(str(sc.get('voiceover', '')))}»</i>"
            for sc in (s1.get("scenes") or [])[:4]
        )

        reply = (
            f"🎯 <b>{h(str(res.get('product_title', '')))}</b>\n\n"
            f"🔥 <b>ТОП-ХУКИ:</b>\n{hooks}\n\n"
            f"🎬 <b>СЦЕНАРИЙ: {h(str(s1.get('title', '')))} ({h(str(s1.get('virality_score', 96)))}% виральность)</b>\n"
            f"<b>Хук:</b> «{h(str(s1.get('hook', '')))}»\n\n"
            f"{scenes_txt}\n\n"
            f"📢 <b>CTA:</b> {h(str(s1.get('cta', '')))}"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Открыть AI-Рерайт и Суфлёр в приложении", web_app=WebAppInfo(url=APP_URL_VERSIONED))]
        ])
        await wait_msg.edit_text(reply[:4000], reply_markup=kb, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Telegram handler error: {e}", exc_info=True)
        try:
            await wait_msg.edit_text(
                f"⚠️ <b>Ошибка:</b> {h(error_text(e, 'ru'))}\n\nЕсли нужна помощь — напиши @onadash",
                parse_mode="HTML"
            )
        except Exception:
            pass


# =====================================================================
# FASTAPI APP
# =====================================================================
async def keep_alive_loop():
    await asyncio.sleep(60)
    while True:
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                await c.get(APP_URL.rstrip("/") + "/health")
        except Exception:
            pass
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


app = FastAPI(title="SNIPER UGC TITANIUM", lifespan=lifespan)


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
            return Response(
                content=r.content,
                media_type=r.headers.get("content-type", "image/jpeg"),
                headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600"}
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/analyze")
async def api_analyze_endpoint(req: AnalyzeRequest):
    lang = "en" if req.lang == "en" else "ru"
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail=ERR[lang]["short_query"])
    try:
        return await run_pipeline(req.query, req.format_type, req.tone, lang)
    except Exception as e:
        logging.error(f"API analyze error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, lang))


@app.post("/api/rewrite-script")
async def api_rewrite_script_endpoint(req: RewriteScriptRequest):
    lang = "en" if req.lang == "en" else "ru"
    try:
        updated = await rewrite_single_script(req)
        return {"script": updated}
    except Exception as e:
        logging.error(f"Rewrite error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, lang))


@app.post("/api/tg-post")
async def api_tg_post_endpoint(req: TgPostRequest):
    lang = "en" if req.lang == "en" else "ru"
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
        raise HTTPException(status_code=503, detail="BOT_TOKEN is not configured")
    lang = "en" if req.lang == "en" else "ru"
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
        logging.error(f"Stars invoice error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stars invoice error: {e}")


# =====================================================================
# OBSIDIAN & TITANIUM SILVER FRONTEND (ZERO GREEN, FULL AVATAR + RU/EN)
# =====================================================================
FRONTEND_UI = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<title>SNIPER // TITANIUM STUDIO</title>
<script src="[https://telegram.org/js/telegram-web-app.js](https://telegram.org/js/telegram-web-app.js)"></script>
<link rel="preconnect" href="[https://fonts.googleapis.com](https://fonts.googleapis.com)">
<link rel="preconnect" href="[https://fonts.gstatic.com](https://fonts.gstatic.com)" crossorigin>
<link href="[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Unbounded:wght@600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Unbounded:wght@600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap)" rel="stylesheet">
<style>
  :root {
    --bg: #050507;
    --card: #0c0c10;
    --card-elevated: #13131a;
    --card-glass: rgba(16, 16, 22, 0.86);
    --border: rgba(255, 255, 255, 0.08);
    --border-light: rgba(255, 255, 255, 0.16);
    --border-focus: rgba(255, 255, 255, 0.42);
    --text: #f8f8fc;
    --muted: #848496;
    --silver: #ffffff;
    --silver-soft: rgba(255, 255, 255, 0.1);
    --silver-glow: rgba(255, 255, 255, 0.18);
    --indigo: #818cf8;
    --gold: #fbbf24;
    --danger: #f43f5e;
    --ok: #38bdf8;
    --radius-xl: 22px;
    --radius-lg: 16px;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --display: 'Unbounded', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; line-height: 1.48; overflow-x: hidden; -webkit-font-smoothing: antialiased; }
  body {
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse 85% 35% at 50% -10%, rgba(255, 255, 255, 0.07), transparent),
      radial-gradient(circle at 95% 20%, rgba(129, 140, 248, 0.05), transparent 45%);
  }
  .app { max-width: 500px; margin: 0 auto; padding: 12px 16px calc(112px + env(safe-area-inset-bottom)); }
  .mono { font-family: var(--mono); }
  button { font-family: inherit; user-select: none; }

  /* HEADER WITH MINI USER AVATAR, BALANCE & RU/EN SWITCH */
  .header { display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 6px 0 16px; }
  .brand-cluster { display: flex; align-items: center; gap: 9px; cursor: pointer; }
  .brand-mark {
    width: 32px; height: 32px; border-radius: 10px;
    background: linear-gradient(145deg, #23232e, #0f0f15);
    border: 1px solid var(--border-light);
    display: flex; align-items: center; justify-content: center;
    font-family: var(--display); font-weight: 800; font-size: 11px; color: #fff;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.2);
  }
  .brand-name { font-family: var(--display); font-weight: 800; font-size: 14px; letter-spacing: 0.03em; color: #fff; }
  .dev-pill {
    display: none; align-items: center; gap: 4px; font-family: var(--mono); font-size: 9px; font-weight: 700;
    color: #000; background: #fff; padding: 3px 7px; border-radius: 6px; box-shadow: 0 0 14px var(--silver-glow);
  }
  .dev-pill.active { display: inline-flex; }

  .top-right { display: flex; align-items: center; gap: 7px; }
  .wallet-pill {
    display: flex; align-items: center; gap: 6px;
    background: var(--card-elevated); border: 1px solid var(--border-light);
    padding: 6px 11px; border-radius: 999px; color: var(--text); font-weight: 700; font-size: 12.5px; cursor: pointer;
    transition: transform 0.15s ease;
  }
  .wallet-pill:active { transform: scale(0.95); }
  .wallet-star { color: var(--gold); font-size: 13px; }

  .lang-sw { display: flex; background: var(--card-elevated); border: 1px solid var(--border-light); border-radius: 999px; padding: 2px; }
  .lang-sw button {
    background: transparent; border: none; color: var(--muted); font-weight: 700; font-size: 10.5px;
    padding: 4px 8px; border-radius: 999px; cursor: pointer; transition: all 0.18s ease;
  }
  .lang-sw button.active { background: #fff; color: #000; }

  /* VIEWS */
  .view { display: none; }
  .view.active { display: block; animation: fadeSlide 0.26s cubic-bezier(0.16, 1, 0.3, 1); }
  @keyframes fadeSlide { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }

  /* BENTO CARDS */
  .bento-card {
    background: var(--card); border: 1px solid var(--border); border-top-color: var(--border-light);
    border-radius: var(--radius-xl); padding: 18px; margin-bottom: 14px;
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.6);
  }
  .section-kicker {
    font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--muted); margin-bottom: 6px; display: flex; align-items: center; gap: 6px;
  }
  .section-kicker span.dot { width: 6px; height: 6px; border-radius: 50%; background: #fff; box-shadow: 0 0 8px #fff; }
  .hero-heading { font-family: var(--display); font-weight: 700; font-size: 18.5px; line-height: 1.25; letter-spacing: -0.02em; margin-bottom: 6px; }
  .hero-sub { color: var(--muted); font-size: 13px; line-height: 1.45; }

  /* COMMAND DOCK */
  .cmd-dock {
    background: #07070a; border: 1.5px solid var(--border-light); border-radius: 18px;
    padding: 12px; margin: 14px 0 12px; transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .cmd-dock:focus-within { border-color: #fff; box-shadow: 0 0 0 4px rgba(255,255,255,0.08); }
  .cmd-input {
    width: 100%; background: transparent; border: none; outline: none; color: var(--text);
    font-family: inherit; font-size: 14px; font-weight: 500; line-height: 1.4; resize: none; min-height: 44px; max-height: 96px;
  }
  .cmd-input::placeholder { color: #525263; font-weight: 400; }
  .cmd-bar { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); gap: 8px; flex-wrap: wrap; }
  .dock-btn {
    background: var(--card-elevated); color: var(--text); border: 1px solid var(--border-light);
    padding: 7px 11px; border-radius: 10px; font-size: 11.5px; font-weight: 600; cursor: pointer;
    display: inline-flex; align-items: center; gap: 5px; transition: all 0.15s ease;
  }
  .dock-btn:active { transform: scale(0.95); background: #fff; color: #000; }
  .params-trigger {
    background: rgba(255, 255, 255, 0.04); border: 1px solid var(--border-light); color: var(--text);
    padding: 7px 12px; border-radius: 10px; font-size: 11.5px; font-weight: 600; cursor: pointer;
    display: inline-flex; align-items: center; gap: 7px;
  }
  .params-dot { width: 6px; height: 6px; border-radius: 50%; background: #fff; }

  /* LIVE PRODUCT OVERVIEW CARD */
  .overview-card {
    display: none; background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.01)), var(--card-elevated);
    border: 1px solid var(--border-focus); border-radius: 16px; padding: 12px; margin-bottom: 14px;
  }
  .ov-flex { display: flex; align-items: center; gap: 12px; }
  .ov-img { width: 56px; height: 56px; border-radius: 12px; object-fit: cover; background: #050508; border: 1px solid var(--border-light); flex-shrink: 0; }
  .ov-info { flex: 1; min-width: 0; }
  .ov-badges { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap; }
  .ov-source { font-family: var(--mono); font-size: 9.5px; font-weight: 700; background: #fff; color: #000; padding: 2px 7px; border-radius: 5px; text-transform: uppercase; }
  .ov-price { font-family: var(--mono); font-size: 10.5px; font-weight: 700; background: rgba(251, 191, 36, 0.16); color: var(--gold); padding: 2px 7px; border-radius: 5px; }
  .ov-title { font-weight: 700; font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .ov-meta { font-size: 11.5px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

  /* BUTTONS (LIQUID PLATINUM WHITE) */
  .btn-primary {
    width: 100%; background: linear-gradient(180deg, #ffffff 0%, #dcdce6 100%);
    color: #050507; border: none; padding: 15px 18px; border-radius: 15px;
    font-size: 13px; font-weight: 800; letter-spacing: 0.02em; cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 8px;
    box-shadow: 0 8px 26px rgba(255, 255, 255, 0.14); transition: transform 0.12s ease, opacity 0.15s ease;
  }
  .btn-primary:active { transform: scale(0.98); }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }
  .btn-secondary {
    width: 100%; background: var(--card-elevated); color: var(--text); border: 1px solid var(--border-light);
    padding: 12px 16px; border-radius: 14px; font-size: 12.5px; font-weight: 700; cursor: pointer;
    display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  .btn-secondary:active { transform: scale(0.98); border-color: #fff; }
  .sub-note { text-align: center; font-family: var(--mono); font-size: 10.5px; color: var(--muted); margin-top: 10px; }

  /* AI NEURAL CONSOLE */
  .neural-box { display: none; background: #08080c; border: 1px solid var(--border-focus); border-radius: var(--radius-xl); padding: 16px; margin-bottom: 16px; }
  .nb-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
  .nb-title { font-family: var(--mono); font-size: 11px; color: #fff; font-weight: 700; display: flex; align-items: center; gap: 6px; }
  .nb-bar-track { width: 100%; height: 5px; background: var(--card-elevated); border-radius: 999px; overflow: hidden; margin-bottom: 12px; }
  .nb-bar-fill { height: 100%; width: 15%; background: linear-gradient(90deg, #fff, var(--indigo)); transition: width 0.4s ease; }
  .nb-log { font-family: var(--mono); font-size: 11.5px; line-height: 1.6; color: #d4d4e0; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 12px; padding: 12px; min-height: 104px; }
  .nb-line { margin-bottom: 5px; display: flex; gap: 8px; }
  .nb-line .arrow { color: #fff; }

  /* RESULTS */
  #resultsArea { display: none; }
  .seg-tabs { display: flex; gap: 4px; overflow-x: auto; scrollbar-width: none; background: var(--card); padding: 5px; border: 1px solid var(--border-light); border-radius: 14px; margin-bottom: 14px; }
  .seg-btn { flex: 1; white-space: nowrap; background: transparent; color: var(--muted); border: none; padding: 8px 10px; font-size: 11px; font-weight: 700; border-radius: 10px; cursor: pointer; }
  .seg-btn.active { background: #fff; color: #050507; }
  .seg-pane { display: none; }
  .seg-pane.active { display: block; animation: fadeSlide 0.22s ease; }

  .script-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid var(--border); }
  .script-badge-row { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
  .mini-tag { font-family: var(--mono); font-size: 10px; padding: 3px 8px; border-radius: 6px; background: var(--card-elevated); border: 1px solid var(--border-light); color: var(--muted); }
  .mini-tag.plat { color: #fff; border-color: var(--border-focus); background: var(--silver-soft); }
  .btn-chip { background: var(--card-elevated); border: 1px solid var(--border-light); color: var(--text); padding: 6px 10px; border-radius: 9px; font-family: var(--mono); font-size: 10.5px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; }
  .btn-chip.accent { background: #fff; color: #000; border-color: #fff; }
  .hook-callout { background: #07070a; border-left: 3px solid #fff; padding: 11px 13px; border-radius: 0 12px 12px 0; margin-bottom: 10px; font-size: 13.5px; font-weight: 600; }
  .scene-block { background: #07070a; border: 1px solid var(--border); border-radius: 13px; padding: 12px; margin-bottom: 8px; font-size: 13px; }
  .scene-time { font-family: var(--mono); font-size: 10.5px; color: var(--indigo); margin-bottom: 4px; }
  .prompt-strip { margin-top: 8px; padding: 8px 10px; background: var(--card); border: 1px dashed var(--border-light); border-radius: 9px; font-family: var(--mono); font-size: 11px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; gap: 8px; }

  /* AI REWRITE BAR */
  .rw-box { margin-top: 14px; padding-top: 12px; border-top: 1px dashed var(--border-light); }
  .rw-title { font-family: var(--mono); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .rw-grid { display: flex; flex-wrap: wrap; gap: 6px; }
  .rw-pill { background: #07070a; border: 1px solid var(--border-light); color: var(--text); padding: 6px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 600; cursor: pointer; }
  .rw-pill:active { background: #fff; color: #000; border-color: #fff; }

  /* TELEPROMPTER */
  .tp-script-pick { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 12px; background: #07070a; border: 1px solid var(--border); border-radius: 13px; margin-bottom: 8px; cursor: pointer; }
  .tp-script-pick.selected { border-color: #fff; background: var(--silver-soft); }
  .tp-editor { width: 100%; min-height: 150px; background: #07070a; border: 1px solid var(--border-light); border-radius: 14px; color: var(--text); padding: 14px; font-family: inherit; font-size: 15px; line-height: 1.55; resize: vertical; outline: none; margin-bottom: 12px; }
  .tp-player { position: fixed; inset: 0; z-index: 600; background: #000; display: none; flex-direction: column; justify-content: space-between; padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom); }
  .tp-player.open { display: flex; }
  .tp-p-top { display: flex; justify-content: space-between; align-items: center; padding: 12px 16px; background: rgba(12, 12, 16, 0.94); border-bottom: 1px solid var(--border); z-index: 5; }
  .tp-p-stage { flex: 1; overflow-y: auto; padding: 36vh 22px 45vh; }
  .tp-p-stage.mirror { transform: scaleX(-1); }
  .tp-eye-band { position: fixed; top: 35%; left: 0; right: 0; height: 92px; pointer-events: none; z-index: 3; border-top: 1.5px dashed rgba(255,255,255,0.45); border-bottom: 1.5px dashed rgba(255,255,255,0.45); background: linear-gradient(90deg, rgba(255,255,255,0.06), transparent 15%, transparent 85%, rgba(255,255,255,0.06)); }
  .tp-p-text { font-size: 28px; font-weight: 800; line-height: 1.45; color: #fff; text-align: center; white-space: pre-wrap; outline: none; }
  .tp-p-controls { background: rgba(12, 12, 16, 0.96); border-top: 1px solid var(--border); padding: 12px 16px 16px; z-index: 5; }
  .tp-cd-overlay { position: fixed; inset: 0; z-index: 10; background: rgba(0,0,0,0.85); display: none; align-items: center; justify-content: center; font-family: var(--display); font-size: 84px; font-weight: 800; color: #fff; }

  /* PROFILE AVATAR & STATS IN CABINET */
  .prof-row { display: flex; align-items: center; gap: 14px; }
  .prof-avatar {
    width: 58px; height: 58px; border-radius: 50%; background: var(--card-elevated);
    border: 1.5px solid var(--border-focus); display: flex; align-items: center; justify-content: center;
    font-family: var(--display); font-size: 20px; font-weight: 800; color: #fff;
    overflow: hidden; flex-shrink: 0; box-shadow: 0 6px 18px rgba(0,0,0,0.5);
  }
  .prof-avatar img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .stats-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .stat-box { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 13px 14px; }
  .stat-val { font-family: var(--display); font-size: 19px; font-weight: 700; color: #fff; }
  .stat-lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }

  /* INPUTS & MODALS */
  .field-inp { width: 100%; background: #07070a; border: 1px solid var(--border-light); border-radius: 12px; color: var(--text); padding: 11px 13px; font-size: 13.5px; font-family: inherit; outline: none; }
  .field-inp:focus { border-color: #fff; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
  .erid-tag-box { background: rgba(255, 255, 255, 0.05); border: 1px dashed var(--border-focus); border-radius: 11px; padding: 9px 12px; font-family: var(--mono); font-size: 11.5px; color: #e2e2f0; margin-top: 8px; }
  .sheet-backdrop { position: fixed; inset: 0; z-index: 400; background: rgba(0, 0, 0, 0.76); backdrop-filter: blur(8px); display: none; align-items: flex-end; justify-content: center; }
  .sheet-backdrop.open { display: flex; }
  .sheet-box { width: 100%; max-width: 500px; max-height: 88vh; overflow-y: auto; background: var(--card-elevated); border: 1px solid var(--border-light); border-radius: 26px 26px 0 0; padding: 18px 18px calc(24px + env(safe-area-inset-bottom)); animation: sheetUp 0.25s cubic-bezier(0.16, 1, 0.3, 1); }
  @keyframes sheetUp { from { transform: translateY(40px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
  .sheet-handle { width: 38px; height: 4px; background: rgba(255,255,255,0.2); border-radius: 999px; margin: 0 auto 14px; }
  .opt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
  .opt-card { background: #07070a; border: 1.5px solid var(--border); border-radius: 14px; padding: 11px 12px; text-align: left; color: var(--text); cursor: pointer; }
  .opt-card.active { border-color: #fff; background: var(--silver-soft); }
  .opt-card .o-title { font-weight: 700; font-size: 13px; margin-bottom: 2px; display: flex; justify-content: space-between; }
  .opt-card .o-desc { font-size: 11px; color: var(--muted); }
  .pay-methods { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; background: #07070a; padding: 5px; border: 1px solid var(--border-light); border-radius: 13px; margin-bottom: 14px; }
  .pm-btn { background: transparent; border: none; color: var(--muted); padding: 9px 4px; border-radius: 9px; font-size: 11.5px; font-weight: 700; cursor: pointer; }
  .pm-btn.active { background: #fff; color: #000; }
  .qr-mock { width: 148px; height: 148px; margin: 10px auto; background: #fff; border-radius: 14px; padding: 10px; display: flex; align-items: center; justify-content: center; }

  /* FLOATING 4-TAB BOTTOM DOCK */
  .bottom-dock { position: fixed; left: 0; right: 0; bottom: 0; z-index: 100; padding: 8px 12px calc(10px + env(safe-area-inset-bottom)); background: linear-gradient(180deg, rgba(5,5,7,0) 0%, rgba(5,5,7,0.95) 40%); pointer-events: none; }
  .dock-inner { pointer-events: auto; max-width: 480px; margin: 0 auto; display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; background: var(--card-glass); backdrop-filter: blur(22px); border: 1px solid var(--border-light); border-radius: 20px; padding: 5px; box-shadow: 0 14px 34px rgba(0,0,0,0.8); }
  .dock-tab { background: transparent; border: none; color: var(--muted); padding: 7px 2px; border-radius: 14px; display: flex; flex-direction: column; align-items: center; gap: 3px; font-size: 10px; font-weight: 700; cursor: pointer; transition: all 0.18s ease; }
  .dock-tab.active { background: #fff; color: #050507; box-shadow: 0 4px 14px var(--silver-glow); }
  .dock-tab svg { width: 19px; height: 19px; stroke: currentColor; fill: none; stroke-width: 2; }

  .toast { position: fixed; bottom: calc(92px + env(safe-area-inset-bottom)); left: 50%; transform: translateX(-50%); background: #fff; color: #000; padding: 9px 18px; border-radius: 999px; font-size: 12px; font-weight: 800; display: none; z-index: 800; white-space: nowrap; box-shadow: 0 8px 25px rgba(0,0,0,0.65); }
</style>
</head>
<body>
<div class="app">

  <!-- HEADER WITH BRAND, BALANCE & RU/EN LANGUAGE SWITCH -->
  <header class="header">
    <div class="brand-cluster" onclick="onSecretLogoTap()">
      <div class="brand-mark">SU</div>
      <div><div class="brand-name">SNIPER</div></div>
      <span id="devBadgeTop" class="dev-pill">⚡ DEV MODE</span>
    </div>
    <div class="top-right">
      <button class="wallet-pill" onclick="switchView('cabinet')">
        <span class="wallet-star">★</span><span id="topBalanceVal">100</span>
      </button>
      <div class="lang-sw">
        <button id="langBtnRu" class="active" onclick="setLanguage('ru')">RU</button>
        <button id="langBtnEn" onclick="setLanguage('en')">EN</button>
      </div>
    </div>
  </header>

  <!-- VIEW 1: STUDIO -->
  <section id="view-studio" class="view active">
    <div class="bento-card">
      <div class="section-kicker"><span class="dot"></span><span data-i18n="kicker_studio">AI CONTENT STUDIO // TITANIUM</span></div>
      <h1 class="hero-heading" data-i18n="studio_h1">Виральные сценарии за 10 секунд</h1>
      <p class="hero-sub" data-i18n="studio_sub">Вставь ссылку с Wildberries, Ozon или любого сайта — карточка товара определится автоматически.</p>
      <div class="cmd-dock">
        <textarea id="mainQueryInp" class="cmd-input" rows="2" data-i18n-ph="inp_ph" placeholder="Ссылка на товар WB / Ozon, артикул или описание..." oninput="onProductInput(this, 'main')"></textarea>
        <div class="cmd-bar">
          <div style="display:flex; gap:6px;">
            <button type="button" class="dock-btn" onclick="pasteFromClipboard('mainQueryInp', 'main')">📋 <span data-i18n="paste">Вставить</span></button>
            <button type="button" id="clearMainBtn" class="dock-btn" style="display:none;" onclick="clearProductInput('mainQueryInp', 'main')">✕</button>
          </div>
          <button type="button" class="params-trigger" onclick="openSettingsSheet()">
            <span class="params-dot"></span><span id="activeParamsSummary">UGC Обзор · Нативный</span><span>⚙️</span>
          </button>
        </div>
      </div>
      <div id="overviewMain" class="overview-card"></div>
      <button id="generateMainBtn" class="btn-primary" onclick="runFullAnalysis()">
        <span data-i18n="btn_create_scripts">⚡ СОЗДАТЬ СЦЕНАРИИ</span><span id="mainBtnPriceTag" class="mono">(100 ★)</span>
      </button>
      <div class="sub-note" id="studioCostHint"></div>
    </div>

    <div id="studioErrorBox" class="bento-card" style="display:none; border-color:var(--danger); color:#fda4af; font-size:13px;"></div>

    <div id="neuralBox" class="neural-box">
      <div class="nb-top"><div class="nb-title"><span>●</span><span data-i18n="ai_working">AI ПРОДЮСЕР АКТИВЕН</span></div><span class="mono" style="font-size:11px; color:#fff;" id="nbPctTxt">15%</span></div>
      <div class="nb-bar-track"><div class="nb-bar-fill" id="nbBarFill"></div></div>
      <div class="nb-log" id="nbLogStream"></div>
    </div>

    <div id="resultsArea">
      <div id="resultProductCard" class="bento-card" style="padding:14px;"></div>
      <div class="seg-tabs">
        <button class="seg-btn active" onclick="switchResultTab('scripts', this)" data-i18n="tab_scripts">Сценарии</button>
        <button class="seg-btn" onclick="switchResultTab('tgposts', this)" data-i18n="tab_tgposts">TG Посты</button>
        <button class="seg-btn" onclick="switchResultTab('hooks', this)" data-i18n="tab_hooks">Топ-Хуки</button>
        <button class="seg-btn" onclick="switchResultTab('broll', this)" data-i18n="tab_broll">AI Видео</button>
        <button class="seg-btn" onclick="switchResultTab('seo', this)" data-i18n="tab_seo">ЦА & SEO</button>
      </div>
      <div id="pane-scripts" class="seg-pane active"></div>
      <div id="pane-tgposts" class="seg-pane"></div>
      <div id="pane-hooks" class="seg-pane"></div>
      <div id="pane-broll" class="seg-pane"></div>
      <div id="pane-seo" class="seg-pane"></div>
    </div>
  </section>

  <!-- VIEW 2: TELEPROMPTER -->
  <section id="view-prompter" class="view">
    <div class="bento-card">
      <div class="section-kicker"><span class="dot"></span><span>TELEPROMPTER STUDIO</span></div>
      <h1 class="hero-heading" data-i18n="tp_h1">Студийный телесуфлёр</h1>
      <p class="hero-sub" style="margin-bottom:14px;" data-i18n="tp_sub">Выбери готовый сценарий из истории генераций ниже или вставь свой текст.</p>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;" data-i18n="tp_hist_lbl">Твои сгенерированные сценарии:</div>
      <div id="prompterHistoryList" style="margin-bottom:14px;"></div>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:6px;" data-i18n="tp_text_lbl">Текст для чтения с камеры:</div>
      <textarea id="prompterEditorInp" class="tp-editor" placeholder="Выбери сценарий выше или вставь сюда свой текст..."></textarea>
      <div class="grid-2" style="margin-bottom:14px;">
        <div style="background:#07070a; border:1px solid var(--border); border-radius:12px; padding:10px 12px;">
          <div class="mono" style="font-size:10px; color:var(--muted);">СКОРОСТЬ: <strong style="color:#fff;" id="tpTabSpeedVal">1.2x</strong></div>
          <input type="range" style="width:100%; accent-color:#fff; margin-top:6px;" min="0.4" max="3.0" step="0.1" value="1.2" oninput="syncTpSpeed(this.value)">
        </div>
        <div style="background:#07070a; border:1px solid var(--border); border-radius:12px; padding:10px 12px;">
          <div class="mono" style="font-size:10px; color:var(--muted);">ШРИФТ: <strong style="color:#fff;" id="tpTabFontVal">28px</strong></div>
          <input type="range" style="width:100%; accent-color:#fff; margin-top:6px;" min="20" max="44" step="2" value="28" oninput="syncTpFont(this.value)">
        </div>
      </div>
      <button class="btn-primary" onclick="launchFullscreenPrompterFromTab()"><span data-i18n="tp_launch">🔴 ЗАПУСТИТЬ СУФЛЁР</span></button>
    </div>
  </section>

  <!-- VIEW 3: TG AD POSTS + ERID + WATERMARK -->
  <section id="view-tgads" class="view">
    <div class="bento-card">
      <div class="section-kicker"><span class="dot"></span><span>TELEGRAM AD & ОРД</span></div>
      <h1 class="hero-heading" data-i18n="tg_h1">Рекламные посты и маркировка</h1>
      <p class="hero-sub" data-i18n="tg_sub">Создаёт 3 нативных поста строго по характеристикам товара + автоматически формирует токен ERID.</p>
      <div class="cmd-dock">
        <textarea id="tgQueryInp" class="cmd-input" rows="2" data-i18n-ph="inp_ph" placeholder="Вставь ссылку на рекламируемый товар WB / Ozon / Сайт..." oninput="onProductInput(this, 'tg')"></textarea>
        <div class="cmd-bar">
          <button type="button" class="dock-btn" onclick="pasteFromClipboard('tgQueryInp', 'tg')">📋 <span data-i18n="paste">Вставить</span></button>
          <button type="button" id="clearTgBtn" class="dock-btn" style="display:none;" onclick="clearProductInput('tgQueryInp', 'tg')">✕</button>
        </div>
      </div>
      <div id="overviewTg" class="overview-card"></div>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin:10px 0 6px;" data-i18n="erid_lbl">Маркировка рекламы (ERID / ОРД ФЗ-38)</div>
      <div class="grid-2">
        <input id="eridOrgInp" class="field-inp" placeholder="Рекламодатель (ООО / ИП + ИНН)" oninput="refreshEridFooters()">
        <input id="eridTokenInp" class="field-inp mono" placeholder="erid: 2Vtzqv..." oninput="refreshEridFooters()">
      </div>
      <input id="eridUrlInp" class="field-inp" placeholder="Партнёрская ссылка или артикул" oninput="refreshEridFooters()" style="margin-bottom:10px;">
      <div id="eridLivePreview" class="erid-tag-box" style="display:none;"></div>
      <button id="tgGenerateBtn" class="btn-primary" style="margin-top:12px;" onclick="runStandaloneTgPosts()">
        <span data-i18n="btn_create_posts">✨ СГЕНЕРИРОВАТЬ 3 ПОСТА</span><span id="tgBtnPriceTag" class="mono">(75 ★)</span>
      </button>
    </div>
    <div id="tgPostsListArea"></div>
    <div class="bento-card">
      <div class="section-kicker"><span class="dot"></span><span>WATERMARK STUDIO</span></div>
      <h2 style="font-size:16px; font-weight:800; margin-bottom:4px;" data-i18n="wm_h2">Водяной знак на фото товара</h2>
      <div class="grid-2" style="margin-top:10px;">
        <input id="wmHandleInp" class="field-inp" placeholder="@ваш_канал" value="@sniper_ugc" oninput="renderWatermarkCanvas()">
        <input id="wmEridStampInp" class="field-inp mono" placeholder="Реклама • erid: ..." oninput="renderWatermarkCanvas()">
      </div>
      <div style="display:flex; gap:6px; margin:8px 0 10px; overflow-x:auto;">
        <button class="btn-chip accent" onclick="setWmPosition('br', this)">Справа внизу</button>
        <button class="btn-chip" onclick="setWmPosition('bl', this)">Слева внизу</button>
        <button class="btn-chip" onclick="setWmPosition('center', this)">По диагонали</button>
        <button class="btn-chip" onclick="setWmPosition('tl', this)">Сверху</button>
      </div>
      <div style="background:#050508; border:1px solid var(--border-light); border-radius:14px; overflow:hidden; margin:12px 0; display:flex; justify-content:center;">
        <canvas id="wmCanvas" width="600" height="750" style="max-width:100%; max-height:300px; display:block;"></canvas>
      </div>
      <div class="grid-2">
        <label class="btn-secondary" style="cursor:pointer;"><span>📷 Своё фото</span><input type="file" accept="image/*" style="display:none;" onchange="onCustomWatermarkFile(event)"></label>
        <button class="btn-primary" onclick="downloadWatermarkPng()"><span>⬇ Скачать PNG</span></button>
      </div>
    </div>
  </section>

  <!-- VIEW 4: CABINET WITH REAL AVATAR, USERNAME, STATS & TRANSACTIONS -->
  <section id="view-cabinet" class="view">
    <div class="bento-card">
      <div id="cabProfileHeader" class="prof-row"></div>
    </div>

    <div id="ownerDevConsole" class="bento-card" style="display:none; border-color:#fff; background:linear-gradient(135deg, rgba(255,255,255,0.1), rgba(12,12,16,0.95));">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span class="mono" style="font-size:11.5px; font-weight:800; color:#fff;">⚡ DEVELOPER MODE // ROOT ACCESS</span>
        <span class="mini-tag plat">OWNER</span>
      </div>
      <p style="font-size:12px; color:#d4d4e0; margin-bottom:10px;">Режим владельца активен: безлимитные генерации (∞) без списания звёзд.</p>
      <div class="grid-2">
        <button class="btn-primary" style="padding:10px;" onclick="devGrantStars(1000)">+1000 ★ ТЕСТ</button>
        <button class="btn-secondary" style="padding:10px;" onclick="setDevModeState(false)">Выключить Dev</button>
      </div>
    </div>

    <div class="bento-card" style="background:linear-gradient(135deg, rgba(255,255,255,0.09), rgba(18,18,26,0.95)); border-color:var(--border-focus);">
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase;" data-i18n="cab_bal_lbl">ТВОЙ БАЛАНС</div>
      <div style="font-family:var(--display); font-size:36px; font-weight:800; color:#fff; line-height:1.15; margin:6px 0 4px;" id="cabBigBalance">100 ★</div>
      <div style="font-size:12px; color:var(--muted); margin-bottom:14px;" data-i18n="cab_rates_lbl">1 сценарий = 100 ★ · 1 пакет TG-постов = 75 ★</div>
      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-bottom:12px;" id="cabPackCards"></div>
      <button class="btn-primary" onclick="openCheckoutSheet(1)"><span data-i18n="cab_topup_btn">💳 ПОПОЛНИТЬ (STARS / СБП / USDT)</span></button>
    </div>

    <div class="stats-row" id="cabStatsGrid"></div>

    <div class="grid-2" style="margin-bottom:14px;">
      <button class="btn-secondary" onclick="openSupportChat()"><span>💬 Саппорт @onadash</span></button>
      <button class="btn-secondary" onclick="shareBotReferral()"><span>🎁 Другу (+50 ★)</span></button>
    </div>

    <div class="bento-card">
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;" data-i18n="promo_lbl">Активация промокода</div>
      <div style="display:flex; gap:8px;">
        <input id="promoCodeInp" class="field-inp mono" placeholder="Промокод (напр. SNIPER)" autocapitalize="characters">
        <button class="btn-primary" style="width:auto; padding:0 18px;" onclick="redeemPromoCode()">ОК</button>
      </div>
    </div>

    <div class="bento-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <span style="font-weight:800; font-size:14px;" data-i18n="hist_lbl">История генераций</span>
        <button style="background:none; border:none; color:var(--muted); font-size:11.5px; text-decoration:underline; cursor:pointer;" onclick="wipeHistory()" data-i18n="clear_btn">Очистить</button>
      </div>
      <div id="cabHistoryContainer"></div>
    </div>

    <div class="bento-card">
      <div style="font-weight:800; font-size:14px; margin-bottom:10px;" data-i18n="tx_lbl">История операций</div>
      <div id="cabTxContainer"></div>
    </div>
  </section>
</div>

<!-- FULLSCREEN TELEPROMPTER PLAYER -->
<div id="tpFullscreenPlayer" class="tp-player">
  <div class="tp-p-top">
    <div class="mono" style="font-size:11.5px; color:#fff; font-weight:700;">🔴 REC // TELEPROMPTER</div>
    <div style="display:flex; gap:6px;">
      <button class="btn-chip" id="tpMirrorToggleBtn" onclick="togglePrompterMirror()">⇄ ЗЕРКАЛО</button>
      <button class="btn-chip" onclick="resetPrompterScroll()">↺ СНАЧАЛА</button>
      <button class="btn-chip" style="border-color:var(--danger); color:var(--danger);" onclick="closeFullscreenPrompter()">✕ ВЫХОД</button>
    </div>
  </div>
  <div class="tp-eye-band"></div>
  <div id="tpCountdownBox" class="tp-cd-overlay">3</div>
  <div id="tpScrollStage" class="tp-p-stage" onclick="togglePrompterPlayPause()">
    <div id="tpLiveTextNode" class="tp-p-text" contenteditable="true" spellcheck="false"></div>
  </div>
  <div class="tp-p-controls">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
      <span class="mono" style="font-size:11px; color:var(--muted); width:95px;">ТЕМП: <strong style="color:#fff;" id="tpModalSpeedLbl">1.2x</strong></span>
      <input type="range" id="tpModalSpeedRange" style="flex:1; accent-color:#fff;" min="0.4" max="3.0" step="0.1" value="1.2" oninput="syncTpSpeed(this.value)">
      <button class="btn-chip" onclick="nudgePrompterFont(-2)">A-</button>
      <button class="btn-chip" onclick="nudgePrompterFont(2)">A+</button>
    </div>
    <button id="tpPlayActionBtn" class="btn-primary" onclick="startPrompterWithCountdown()">▶ СТАРТ С ОТСЧЁТОМ (3 СЕК)</button>
  </div>
</div>

<!-- BOTTOM SHEET -->
<div id="sheetBackdrop" class="sheet-backdrop" onclick="if(event.target===this)closeSheet()"><div id="sheetContent" class="sheet-box"></div></div>

<!-- 4-TAB FLOATING DOCK -->
<nav class="bottom-dock">
  <div class="dock-inner">
    <button id="tabBtn-studio" class="dock-tab active" onclick="switchView('studio')">
      <svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z"/></svg>
      <span data-i18n="nav_studio">Сценарии</span>
    </button>
    <button id="tabBtn-prompter" class="dock-tab" onclick="switchView('prompter')">
      <svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="14" rx="2"/><path d="M8 21h8"/><path d="M12 18v3"/></svg>
      <span data-i18n="nav_prompter">Суфлёр</span>
    </button>
    <button id="tabBtn-tgads" class="dock-tab" onclick="switchView('tgads')">
      <svg viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
      <span data-i18n="nav_tgads">TG Посты</span>
    </button>
    <button id="tabBtn-cabinet" class="dock-tab" onclick="switchView('cabinet')">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></svg>
      <span data-i18n="nav_cabinet">Кабинет</span>
    </button>
  </div>
</nav>

<div id="toastBanner" class="toast"></div>

<script>
  const SERVER_ADMIN_ID = "__ADMIN_TG_ID__";
  const SUPPORT_USER = "onadash";
  const tg = (window.Telegram && window.Telegram.WebApp) ? window.Telegram.WebApp : null;
  if (tg) {
    try {
      tg.ready(); tg.expand();
      if (tg.setHeaderColor) tg.setHeaderColor('#050507');
      if (tg.setBackgroundColor) tg.setBackgroundColor('#050507');
    } catch (e) {}
  }
  const tgUser = (tg && tg.initDataUnsafe && tg.initDataUnsafe.user) ? tg.initDataUnsafe.user : null;

  function isOwnerAccount() {
    if (!tgUser) return false;
    if (SERVER_ADMIN_ID && String(tgUser.id) === String(SERVER_ADMIN_ID)) return true;
    const name = [tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ').toLowerCase();
    const uname = (tgUser.username || '').toLowerCase();
    return (name.includes('aleksandr') && name.includes('druzhinin')) || (name.includes('александр') && name.includes('дружинин')) || uname.includes('druzhinin') || uname.includes('onadash');
  }

  const I18N = {
    ru: {
      nav_studio: 'Сценарии', nav_prompter: 'Суфлёр', nav_tgads: 'TG Посты', nav_cabinet: 'Кабинет',
      kicker_studio: 'AI CONTENT STUDIO // TITANIUM',
      studio_h1: 'Виральные сценарии за 10 секунд',
      studio_sub: 'Вставь ссылку с Wildberries, Ozon или любого сайта — карточка товара определится автоматически.',
      inp_ph: 'Ссылка на товар WB / Ozon, артикул или описание...',
      paste: 'Вставить',
      btn_create_scripts: '⚡ СОЗДАТЬ СЦЕНАРИИ',
      ai_working: 'AI ПРОДЮСЕР АКТИВЕН',
      tab_scripts: 'Сценарии', tab_tgposts: 'TG Посты', tab_hooks: 'Топ-Хуки', tab_broll: 'AI Видео', tab_seo: 'ЦА & SEO',
      tp_h1: 'Студийный телесуфлёр',
      tp_sub: 'Выбери готовый сценарий из истории генераций ниже или вставь свой текст.',
      tp_hist_lbl: 'Твои сгенерированные сценарии:',
      tp_text_lbl: 'Текст для чтения с камеры:',
      tp_launch: '🔴 ЗАПУСТИТЬ СУФЛЁР',
      tg_h1: 'Рекламные посты и маркировка',
      tg_sub: 'Создаёт 3 нативных поста строго по характеристикам товара + автоматически формирует токен ERID.',
      erid_lbl: 'Маркировка рекламы (ERID / ОРД ФЗ-38)',
      btn_create_posts: '✨ СГЕНЕРИРОВАТЬ 3 ПОСТА',
      wm_h2: 'Водяной знак на фото товара',
      cab_bal_lbl: 'ТВОЙ БАЛАНС ЗВЁЗД',
      cab_rates_lbl: '1 сценарий = 100 ★ · 1 пакет TG-постов = 75 ★',
      cab_topup_btn: '💳 ПОПОЛНИТЬ (STARS / СБП / USDT)',
      promo_lbl: 'Активация промокода',
      hist_lbl: 'История генераций',
      tx_lbl: 'История операций',
      clear_btn: 'Очистить',
      stat_gens: 'Создано проектов',
      stat_spent: 'Потрачено ★',
      tx_bonus: 'Приветственный бонус',
      tx_topup: 'Пополнение баланса',
      tx_gen: 'Генерация сценария',
      tx_tg: 'Генерация TG-постов',
      tx_promo: 'Промокод'
    },
    en: {
      nav_studio: 'Scripts', nav_prompter: 'Prompter', nav_tgads: 'TG Ads', nav_cabinet: 'Account',
      kicker_studio: 'AI CONTENT STUDIO // TITANIUM',
      studio_h1: 'Viral Scripts in 10 Seconds',
      studio_sub: 'Paste any Wildberries, Ozon or website link — product card is detected automatically.',
      inp_ph: 'Paste WB / Ozon / website link, SKU or description...',
      paste: 'Paste',
      btn_create_scripts: '⚡ GENERATE SCRIPTS',
      ai_working: 'AI PRODUCER ACTIVE',
      tab_scripts: 'Scripts', tab_tgposts: 'TG Posts', tab_hooks: 'Top Hooks', tab_broll: 'AI Video', tab_seo: 'Audience & SEO',
      tp_h1: 'Studio Teleprompter',
      tp_sub: 'Pick a generated script from your history below or paste custom text.',
      tp_hist_lbl: 'Your generated scripts:',
      tp_text_lbl: 'Camera script text:',
      tp_launch: '🔴 LAUNCH TELEPROMPTER',
      tg_h1: 'Telegram Ad Posts & ERID',
      tg_sub: 'Creates 3 native blogger posts strictly matching product specs + legal ERID disclosure.',
      erid_lbl: 'Ad Disclosure (ERID / Legal)',
      btn_create_posts: '✨ GENERATE 3 AD POSTS',
      wm_h2: 'Product Photo Watermark',
      cab_bal_lbl: 'YOUR STARS BALANCE',
      cab_rates_lbl: '1 script = 100 ★ · 1 TG post pack = 75 ★',
      cab_topup_btn: '💳 TOP UP (STARS / CARD / USDT)',
      promo_lbl: 'Redeem Promo Code',
      hist_lbl: 'Generation History',
      tx_lbl: 'Transaction Log',
      clear_btn: 'Clear',
      stat_gens: 'Projects created',
      stat_spent: 'Stars spent ★',
      tx_bonus: 'Welcome bonus',
      tx_topup: 'Balance top-up',
      tx_gen: 'Script generation',
      tx_tg: 'TG Post generation',
      tx_promo: 'Promo code'
    }
  };

  const STORE_KEY = 'sniper_titanium_v15';
  const SCRIPT_PRICE = 100, TG_POST_PRICE = 75, START_BONUS = 100;
  const PROMOS = { 'SNIPER': 100, 'UGC2026': 150 };
  const PACKAGES = [
    { id: 'pack_100', credits: 100, stars: 100, rub: '190 ₽', usdt: '2.0 USDT', tag: '1 сценарий' },
    { id: 'pack_300', credits: 300, stars: 300, rub: '490 ₽', usdt: '5.2 USDT', tag: 'ХИТ ВЫБОРА', hot: true },
    { id: 'pack_1000', credits: 1000, stars: 850, rub: '1 390 ₽', usdt: '14.5 USDT', tag: 'PRO -15%' }
  ];
  const FORMAT_OPTIONS = [
    { id: 'ugc', title: 'UGC Обзор', desc: 'Живой нативный отзыв покупателя' },
    { id: 'problem_solution', title: 'Боль → Решение', desc: 'Острая боль ЦА и решение товаром' },
    { id: 'unboxing', title: 'ASMR Распаковка', desc: 'Эффект первого впечатления' },
    { id: 'aesthetic', title: 'Aesthetic Визуал', desc: 'Минимализм под трендовый звук' },
    { id: 'storytelling', title: 'Сторителлинг', desc: 'История с интригой в начале' }
  ];
  const TONE_OPTIONS = [
    { id: 'native', title: 'Нативный', desc: 'Искренне, как другу в голосовом' },
    { id: 'aggressive', title: 'Дерзкий', desc: 'Хлёсткие триггеры и удержание' },
    { id: 'humor', title: 'С юмором', desc: 'Жизненная ирония и лёгкость' },
    { id: 'luxury', title: 'Премиум', desc: 'Спокойный дорогой стиль' }
  ];

  function createDefaultState() {
    return {
      lang: 'ru',
      devMode: isOwnerAccount(),
      balance: START_BONUS,
      gens: 0,
      spent: 0,
      promoUsed: false,
      format: 'ugc',
      tone: 'native',
      history: [],
      tx: [{ type: 'bonus', amount: START_BONUS, ts: Date.now() }]
    };
  }

  function loadAppState() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const p = JSON.parse(raw);
        const st = Object.assign(createDefaultState(), p);
        if (isOwnerAccount() && p.devMode === undefined) st.devMode = true;
        if (!Array.isArray(st.tx)) st.tx = [{ type: 'bonus', amount: START_BONUS, ts: Date.now() }];
        return st;
      }
    } catch (e) {}
    return createDefaultState();
  }
  function saveAppState() { try { localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch (e) {} }

  let S = loadAppState();
  if (!I18N[S.lang]) S.lang = 'ru';
  let lastResultData = null, livePreviewCache = null;
  const copyVault = [], prompterVault = [];

  function t(k) { return (I18N[S.lang] && I18N[S.lang][k]) || I18N.ru[k] || k; }
  function esc(s) { return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function saveCopy(txt) { copyVault.push(String(txt || '')); return copyVault.length - 1; }
  function copyVaultIdx(i) { copyText(copyVault[i]); }
  function savePrompter(txt) { prompterVault.push(String(txt || '')); return prompterVault.length - 1; }
  function logTx(type, amount) { S.tx.unshift({ type: type, amount: amount, ts: Date.now() }); S.tx = S.tx.slice(0, 30); }

  let toastTimer = null;
  function toast(msg) {
    const el = document.getElementById('toastBanner');
    el.textContent = msg; el.style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.style.display = 'none'; }, 2000);
  }
  function haptic(type) { try { if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred(type || 'success'); } catch (e) {} }
  function copyText(txt) {
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(txt).catch(() => {});
    toast(S.lang === 'en' ? 'Copied!' : 'Скопировано в буфер'); haptic('success');
  }
  function openSupportChat() { const u = '[https://t.me/](https://t.me/)' + SUPPORT_USER; tg?.openTelegramLink ? tg.openTelegramLink(u) : window.open(u, '_blank'); }
  function shareBotReferral() {
    const u = '[https://t.me/share/url?url=](https://t.me/share/url?url=)' + encodeURIComponent('[https://t.me/sniperugc_bot](https://t.me/sniperugc_bot)') + '&text=' + encodeURIComponent('🔥 AI-студия виральных сценариев для WB/Ozon с телесуфлёром! Забирай 100 ★:');
    tg?.openTelegramLink ? tg.openTelegramLink(u) : window.open(u, '_blank');
  }

  function setLanguage(lang) {
    if (!I18N[lang]) return;
    S.lang = lang;
    saveAppState();
    applyTranslations();
  }

  function applyTranslations() {
    document.getElementById('langBtnRu').classList.toggle('active', S.lang === 'ru');
    document.getElementById('langBtnEn').classList.toggle('active', S.lang === 'en');
    document.querySelectorAll('[data-i18n]').forEach(el => { el.textContent = t(el.dataset.i18n); });
    document.querySelectorAll('[data-i18n-ph]').forEach(el => { el.placeholder = t(el.dataset.i18nPh); });
    syncHeaderAndPrices();
    renderCabinetView();
  }

  let logoTaps = 0, logoTapTimer = null;
  function onSecretLogoTap() {
    logoTaps++; clearTimeout(logoTapTimer);
    if (logoTaps >= 5) { logoTaps = 0; setDevModeState(!S.devMode); }
    else logoTapTimer = setTimeout(() => { logoTaps = 0; }, 1400);
  }
  function setDevModeState(act) {
    S.devMode = !!act; saveAppState(); syncHeaderAndPrices(); renderCabinetView();
    toast(S.devMode ? '⚡ DEVELOPER MODE АКТИВИРОВАН' : 'Developer Mode отключён'); haptic('success');
  }
  function devGrantStars(a) { S.balance += a; logTx('topup', a); saveAppState(); syncHeaderAndPrices(); renderCabinetView(); toast(`+${a} ★ начислено!`); }

  function syncHeaderAndPrices() {
    document.getElementById('devBadgeTop').classList.toggle('active', !!S.devMode);
    document.getElementById('topBalanceVal').textContent = S.devMode ? '∞ DEV' : S.balance;
    document.getElementById('mainBtnPriceTag').textContent = S.devMode ? '(DEV ∞)' : `(${SCRIPT_PRICE} ★)`;
    document.getElementById('tgBtnPriceTag').textContent = S.devMode ? '(DEV ∞)' : `(${TG_POST_PRICE} ★)`;
    document.getElementById('studioCostHint').textContent = S.devMode
      ? '⚡ DEVELOPER MODE: безлимитная генерация сценариев и постов (0 ★)'
      : (S.lang === 'en' ? `Cost: ${SCRIPT_PRICE} ★ · AI rewrites inside are free` : `Стоимость: ${SCRIPT_PRICE} ★ · Все AI-рерайты внутри сценария бесплатны`);
    const fmt = FORMAT_OPTIONS.find(x => x.id === S.format) || FORMAT_OPTIONS[0];
    const tone = TONE_OPTIONS.find(x => x.id === S.tone) || TONE_OPTIONS[0];
    document.getElementById('activeParamsSummary').textContent = `${fmt.title} · ${tone.title}`;
  }

  function switchView(name) {
    ['studio', 'prompter', 'tgads', 'cabinet'].forEach(v => {
      document.getElementById('view-' + v).classList.toggle('active', v === name);
      document.getElementById('tabBtn-' + v).classList.toggle('active', v === name);
    });
    if (name === 'prompter') renderPrompterTab();
    if (name === 'tgads') renderWatermarkCanvas();
    if (name === 'cabinet') renderCabinetView();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function switchResultTab(name, btn) {
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.seg-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('pane-' + name).classList.add('active');
  }

  function openSheet(h) { document.getElementById('sheetContent').innerHTML = `<div class="sheet-handle"></div>` + h; document.getElementById('sheetBackdrop').classList.add('open'); }
  function closeSheet() { document.getElementById('sheetBackdrop').classList.remove('open'); }

  function openSettingsSheet() {
    const fCards = FORMAT_OPTIONS.map(f => `
      <button type="button" class="opt-card ${S.format === f.id ? 'active' : ''}" onclick="selectFormat('${f.id}')">
        <div class="o-title"><span>${esc(f.title)}</span><span>${S.format === f.id ? '✓' : ''}</span></div>
        <div class="o-desc">${esc(f.desc)}</div>
      </button>`).join('');
    const tCards = TONE_OPTIONS.map(t => `
      <button type="button" class="opt-card ${S.tone === t.id ? 'active' : ''}" onclick="selectTone('${t.id}')">
        <div class="o-title"><span>${esc(t.title)}</span><span>${S.tone === t.id ? '✓' : ''}</span></div>
        <div class="o-desc">${esc(t.desc)}</div>
      </button>`).join('');
    openSheet(`
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;"><div style="font-weight:800; font-size:17px;">⚙️ Параметры сценария</div><button class="btn-chip" onclick="closeSheet()">Готово</button></div>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">01 / Формат ролика</div><div class="opt-grid">${fCards}</div>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">02 / Тональность</div><div class="opt-grid">${tCards}</div>
      <button class="btn-primary" onclick="closeSheet()">ПРИМЕНИТЬ</button>
    `);
  }
  function selectFormat(id) { S.format = id; saveAppState(); syncHeaderAndPrices(); openSettingsSheet(); haptic('success'); }
  function selectTone(id) { S.tone = id; saveAppState(); syncHeaderAndPrices(); openSettingsSheet(); haptic('success'); }

  let previewTimers = {};
  function onProductInput(el, mode) {
    const val = (el.value || '').trim();
    const btn = document.getElementById(mode === 'main' ? 'clearMainBtn' : 'clearTgBtn');
    if (btn) btn.style.display = val.length > 0 ? 'inline-flex' : 'none';
    const mirror = document.getElementById(mode === 'main' ? 'tgQueryInp' : 'mainQueryInp');
    if (mirror && !mirror.value.trim()) mirror.value = val;
    clearTimeout(previewTimers[mode]);
    const card = document.getElementById(mode === 'main' ? 'overviewMain' : 'overviewTg');
    if (val.length < 5) { if (card) card.style.display = 'none'; return; }
    previewTimers[mode] = setTimeout(() => fetchPreview(val, mode), 380);
  }
  async function pasteFromClipboard(id, mode) {
    if (navigator.clipboard?.readText) {
      const clip = await navigator.clipboard.readText().catch(() => '');
      if (clip) { const el = document.getElementById(id); el.value = clip.trim(); onProductInput(el, mode); haptic('success'); return; }
    }
    document.getElementById(id).focus();
  }
  function clearProductInput(id, mode) {
    const el = document.getElementById(id); el.value = ''; onProductInput(el, mode);
    const c = document.getElementById(mode === 'main' ? 'overviewMain' : 'overviewTg');
    if (c) c.style.display = 'none';
  }
  async function fetchPreview(q, mode) {
    const card = document.getElementById(mode === 'main' ? 'overviewMain' : 'overviewTg');
    if (!card) return;
    try {
      const resp = await fetch('/api/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q }) });
      if (!resp.ok) return;
      const d = await resp.json(); livePreviewCache = d;
      card.innerHTML = `
        <div class="ov-flex">
          ${d.image ? `<img class="ov-img" src="${esc(d.image)}" onerror="this.style.display='none'">` : ''}
          <div class="ov-info">
            <div class="ov-badges"><span class="ov-source">✓ ${esc(d.source)}</span>${d.price ? `<span class="ov-price">${esc(d.price)}</span>` : ''}</div>
            <div class="ov-title">${esc(d.title)}</div><div class="ov-meta">${esc([d.category, d.brand].filter(Boolean).join(' · '))}</div>
          </div>
        </div>`;
      card.style.display = 'block';
      if (d.image) loadWatermarkImg(d.image);
    } catch (e) {}
  }

  let neuralTimer = null;
  function startNeuralConsole(q) {
    const b = document.getElementById('neuralBox'), bar = document.getElementById('nbBarFill'), pct = document.getElementById('nbPctTxt'), log = document.getElementById('nbLogStream');
    b.style.display = 'block'; log.innerHTML = '';
    const title = (livePreviewCache?.title || q).slice(0, 40);
    const steps = [
      { p: 20, t: `Загрузка карточки: «${title}»` },
      { p: 45, t: `Фильтрация чужих рекомендаций и выделение характеристик...` },
      { p: 70, t: `Генерация 5 триггерных хуков (удержание 0-2 сек)...` },
      { p: 95, t: `Сборка раскадровки, текста суфлёра и 3 TG-постов...` }
    ];
    let i = 0;
    function run() {
      if (i >= steps.length) return;
      bar.style.width = steps[i].p + '%'; pct.textContent = steps[i].p + '%';
      log.innerHTML += `<div class="nb-line"><span class="arrow">▸</span><span>${esc(steps[i].t)}</span></div>`;
      i++;
    }
    run(); clearInterval(neuralTimer); neuralTimer = setInterval(run, 2100);
  }
  function stopNeuralConsole() { clearInterval(neuralTimer); document.getElementById('neuralBox').style.display = 'none'; }

  async function runFullAnalysis() {
    const err = document.getElementById('studioErrorBox'); err.style.display = 'none';
    const q = document.getElementById('mainQueryInp').value.trim();
    if (!q) { toast('Вставь ссылку на товар'); haptic('error'); return; }
    if (!S.devMode && S.balance < SCRIPT_PRICE) { openCheckoutSheet(0); return; }
    const btn = document.getElementById('generateMainBtn'), res = document.getElementById('resultsArea');
    btn.disabled = true; res.style.display = 'none'; startNeuralConsole(q);
    try {
      const resp = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, format_type: S.format, tone: S.tone, lang: S.lang }) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Ошибка генерации');
      if (!S.devMode) { S.balance -= SCRIPT_PRICE; S.spent += SCRIPT_PRICE; logTx('gen', -SCRIPT_PRICE); }
      S.gens++;
      S.history.unshift({ id: Date.now(), ts: Date.now(), title: data.product_title || q.slice(0, 35), query: q.slice(0, 50), data: data });
      S.history = S.history.slice(0, 15);
      saveAppState(); syncHeaderAndPrices();
      lastResultData = data; renderResults(data);
      res.style.display = 'block';
      if (data.meta_image) loadWatermarkImg(data.meta_image);
      haptic('success');
      setTimeout(() => res.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } catch (e) {
      err.textContent = '⚠️ ' + (e.message || 'Ошибка генерации'); err.style.display = 'block'; haptic('error');
    } finally {
      stopNeuralConsole(); btn.disabled = false;
    }
  }

  function renderResults(d) {
    copyVault.length = 0; prompterVault.length = 0;
    document.getElementById('resultProductCard').innerHTML = `
      <div class="ov-flex">
        ${d.meta_image ? `<img class="ov-img" src="${esc(d.meta_image)}" onerror="this.style.display='none'">` : ''}
        <div class="ov-info">
          <div style="font-weight:800; font-size:15px; margin-bottom:4px;">${esc(d.product_title)}</div>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <span class="mini-tag">${esc(d.meta_source || 'Товар')}</span><span class="mini-tag">${esc(d.category || '')}</span>
            ${d.meta_price ? `<span class="mini-tag plat">${esc(d.meta_price)}</span>` : ''}
          </div>
        </div>
      </div>`;

    document.getElementById('pane-scripts').innerHTML = (d.scripts || []).map((sc, i) => {
      const full = `СЦЕНАРИЙ: ${sc.title}\nХУК: ${sc.hook}\n` + (sc.scenes || []).map(s => `[${s.timecode}] ${s.visual} | Голос: ${s.voiceover}`).join('\n') + `\nCTA: ${sc.cta}`;
      const voice = [sc.hook, ...(sc.scenes || []).map(s => s.voiceover), sc.cta].filter(Boolean).join('\n\n');
      const tpIdx = savePrompter(voice), cpIdx = saveCopy(full);
      const scenes = (sc.scenes || []).map(s => `
        <div class="scene-block">
          <div class="scene-time">⏱ ${esc(s.timecode)}</div>
          <div><strong>Кадр:</strong> ${esc(s.visual)}</div>
          <div style="margin:4px 0; color:#fff;"><strong>Озвучка:</strong> «${esc(s.voiceover)}»</div>
          <div><strong>Плашка:</strong> ${esc(s.on_screen_text)}</div>
          <div class="prompt-strip"><span>🎥 AI: ${esc(s.ai_video_prompt)}</span><button class="btn-chip" onclick="copyVaultIdx(${saveCopy(s.ai_video_prompt)})">COPY</button></div>
        </div>`).join('');
      return `
        <div class="bento-card">
          <div class="script-head">
            <div>
              <div style="font-weight:800; font-size:15px;">0${i+1} // ${esc(sc.title)}</div>
              <div class="script-badge-row"><span class="mini-tag plat">${sc.virality_score || 96}% виральность</span><span class="mini-tag">${esc(sc.angle)}</span></div>
            </div>
            <div style="display:flex; gap:6px;">
              <button class="btn-chip accent" onclick="openPrompterVault(${tpIdx})">📟 В СУФЛЁР</button>
              <button class="btn-chip" onclick="copyVaultIdx(${cpIdx})">📋</button>
            </div>
          </div>
          <div class="hook-callout">«${esc(sc.hook)}»</div>
          ${scenes}
          <div style="font-size:13px; margin-top:8px;"><strong>Призыв (CTA):</strong> ${esc(sc.cta)}</div>
          <div class="rw-box">
            <div class="rw-title">✨ Мгновенный AI-рерайт сценария (Бесплатно):</div>
            <div class="rw-grid">
              <button class="rw-pill" onclick="triggerRewrite(${i}, 'shorter')">✂️ Короче</button>
              <button class="rw-pill" onclick="triggerRewrite(${i}, 'longer')">📝 Подробнее</button>
              <button class="rw-pill" onclick="triggerRewrite(${i}, 'bolder')">🔥 Дерзко</button>
              <button class="rw-pill" onclick="triggerRewrite(${i}, 'native')">💬 Нативнее</button>
              <button class="rw-pill" onclick="triggerRewrite(${i}, 'funny')">😂 Юмор</button>
              <button class="rw-pill" onclick="toggleCustomRw(${i})">✍️ Своя правка</button>
            </div>
            <div id="customRwBox-${i}" style="display:none; gap:6px; margin-top:8px;">
              <input id="customRwInp-${i}" class="field-inp" placeholder="Что изменить в сценарии?">
              <button class="btn-primary" style="width:auto; padding:0 14px;" onclick="triggerRewrite(${i}, 'custom')">ОК</button>
            </div>
          </div>
        </div>`;
    }).join('');

    document.getElementById('pane-tgposts').innerHTML = renderTgPostsHtml(d.tg_posts || []);
    document.getElementById('pane-hooks').innerHTML = `<div class="bento-card">${(d.viral_hooks || []).map((h, idx) => `<div style="display:flex; justify-content:space-between; align-items:center; gap:8px; padding:8px 0; border-bottom:1px solid var(--border);"><div><strong>#0${idx+1}</strong> «${esc(h)}»</div><button class="btn-chip" onclick="copyVaultIdx(${saveCopy(h)})">COPY</button></div>`).join('')}</div>`;
    document.getElementById('pane-broll').innerHTML = `<div class="bento-card">${(d.video_generator_prompts || []).map(p => `<div class="prompt-strip" style="margin-bottom:8px;"><span>${esc(p)}</span><button class="btn-chip" onclick="copyVaultIdx(${saveCopy(p)})">COPY</button></div>`).join('')}</div>`;
    document.getElementById('pane-seo').innerHTML = `
      <div class="bento-card">
        <div style="font-weight:800; margin-bottom:6px;">Целевая аудитория</div><div style="font-size:13px; color:#d4d4e0; margin-bottom:10px;">${esc(d.target_audience)}</div>
        <div style="font-weight:800; margin-bottom:6px;">Преимущества</div><ul style="padding-left:18px; font-size:13px; color:#d4d4e0; margin-bottom:10px;">${(d.killer_features || []).map(f => `<li>${esc(f)}</li>`).join('')}</ul>
        <div style="font-size:13px; margin-bottom:6px;">${esc(d.seo_caption)}</div><div class="mono" style="font-size:11px; color:var(--muted);">${esc((d.hashtags || []).join(' '))}</div>
      </div>`;
  }

  function toggleCustomRw(i) {
    const b = document.getElementById('customRwBox-' + i);
    if (b) b.style.display = b.style.display === 'flex' ? 'none' : 'flex';
  }

  async function triggerRewrite(idx, act) {
    if (!lastResultData?.scripts?.[idx]) return;
    let customTxt = '';
    if (act === 'custom') {
      customTxt = (document.getElementById('customRwInp-' + idx)?.value || '').trim();
      if (!customTxt) { toast('Напиши правку'); return; }
    }
    toast('⚡ AI переписывает сценарий...'); haptic('warning');
    try {
      const resp = await fetch('/api/rewrite-script', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ product_title: lastResultData.product_title || '', script: lastResultData.scripts[idx], action: act, custom_instruction: customTxt, lang: S.lang }) });
      const data = await resp.json();
      if (!resp.ok || !data.script) throw new Error(data.detail || 'Ошибка');
      lastResultData.scripts[idx] = data.script; renderResults(lastResultData);
      toast('✨ Сценарий обновлён'); haptic('success');
    } catch (e) { toast(e.message || 'Ошибка'); }
  }

  // Teleprompter
  let tpSpeed = 1.2, tpFont = 28, tpMirror = false, tpPlaying = false, tpRafId = null, tpLastTime = 0;
  function renderPrompterTab() {
    const el = document.getElementById('prompterHistoryList'), list = [];
    S.history.forEach(h => { (h.data?.scripts || []).forEach(sc => { list.push({ title: `${h.title} — ${sc.title}`, text: [sc.hook, ...(sc.scenes || []).map(x => x.voiceover), sc.cta].filter(Boolean).join('\n\n') }); }); });
    if (!list.length) { el.innerHTML = `<div style="padding:10px; border:1px dashed var(--border-light); border-radius:12px; color:var(--muted); font-size:12px; text-align:center;">История пуста. Создай сценарий в первой вкладке.</div>`; return; }
    el.innerHTML = list.slice(0, 6).map(it => `<div class="tp-script-pick" onclick="loadPrompterEditor('${savePrompter(it.text)}')"><div style="font-weight:700; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">🎬 ${esc(it.title)}</div><span class="btn-chip accent">Выбрать</span></div>`).join('');
    if (!document.getElementById('prompterEditorInp').value.trim() && list[0]) document.getElementById('prompterEditorInp').value = list[0].text;
  }
  function loadPrompterEditor(vIdx) { document.getElementById('prompterEditorInp').value = prompterVault[vIdx] || ''; toast('Текст загружен'); }
  function openPrompterVault(vIdx) { document.getElementById('prompterEditorInp').value = prompterVault[vIdx] || ''; openPrompterModal(prompterVault[vIdx] || ''); }
  function launchFullscreenPrompterFromTab() {
    const txt = document.getElementById('prompterEditorInp').value.trim();
    if (!txt) { toast('Вставь текст в суфлёр'); return; }
    openPrompterModal(txt);
  }
  function syncTpSpeed(v) { tpSpeed = parseFloat(v) || 1.2; document.getElementById('tpTabSpeedVal').textContent = tpSpeed.toFixed(1) + 'x'; document.getElementById('tpModalSpeedLbl').textContent = tpSpeed.toFixed(1) + 'x'; document.getElementById('tpModalSpeedRange').value = tpSpeed; }
  function syncTpFont(v) { tpFont = parseInt(v, 10) || 28; document.getElementById('tpTabFontVal').textContent = tpFont + 'px'; document.getElementById('tpLiveTextNode').style.fontSize = tpFont + 'px'; }
  function nudgePrompterFont(d) { syncTpFont(Math.min(46, Math.max(18, tpFont + d))); }
  function openPrompterModal(txt) {
    stopPrompter(); const node = document.getElementById('tpLiveTextNode');
    node.textContent = txt; node.style.fontSize = tpFont + 'px';
    document.getElementById('tpScrollStage').scrollTop = 0; document.getElementById('tpFullscreenPlayer').classList.add('open');
  }
  function closeFullscreenPrompter() { stopPrompter(); document.getElementById('tpFullscreenPlayer').classList.remove('open'); }
  function resetPrompterScroll() { stopPrompter(); document.getElementById('tpScrollStage').scrollTop = 0; }
  function togglePrompterMirror() { tpMirror = !tpMirror; document.getElementById('tpScrollStage').classList.toggle('mirror', tpMirror); document.getElementById('tpMirrorToggleBtn').classList.toggle('accent', tpMirror); }
  function startPrompterWithCountdown() {
    if (tpPlaying) { stopPrompter(); return; }
    const cd = document.getElementById('tpCountdownBox'); let c = 3; cd.textContent = c; cd.style.display = 'flex';
    const iv = setInterval(() => { c--; if (c > 0) { cd.textContent = c; haptic('warning'); } else { clearInterval(iv); cd.style.display = 'none'; beginScroll(); } }, 800);
  }
  function togglePrompterPlayPause() { if (document.activeElement === document.getElementById('tpLiveTextNode')) return; tpPlaying ? stopPrompter() : beginScroll(); }
  function beginScroll() {
    tpPlaying = true; tpLastTime = performance.now(); document.getElementById('tpPlayActionBtn').textContent = '⏸ ПАУЗА';
    const stage = document.getElementById('tpScrollStage');
    function step(now) {
      if (!tpPlaying) return;
      stage.scrollTop += (40 * tpSpeed) * ((now - tpLastTime) / 1000);
      tpLastTime = now;
      if (stage.scrollTop + stage.clientHeight >= stage.scrollHeight - 4) { stopPrompter(); return; }
      tpRafId = requestAnimationFrame(step);
    }
    tpRafId = requestAnimationFrame(step);
  }
  function stopPrompter() { tpPlaying = false; if (tpRafId) cancelAnimationFrame(tpRafId); const b = document.getElementById('tpPlayActionBtn'); if (b) b.textContent = '▶ СТАРТ / ПРОДОЛЖИТЬ'; }

  // TG Ads & Watermark
  function buildErid() {
    const o = (document.getElementById('eridOrgInp')?.value || '').trim(), tok = (document.getElementById('eridTokenInp')?.value || '').trim().replace(/^erid:\s*/i, ''), l = (document.getElementById('eridUrlInp')?.value || '').trim();
    const p = []; if (l) p.push(`👉 Артикул/Ссылка: ${l}`); if (o || tok) p.push(`Реклама. ${o || 'Рекламодатель'}${tok ? `, erid: ${tok}` : ''}`);
    return p.join('\n');
  }
  function refreshEridFooters() {
    const f = buildErid(), b = document.getElementById('eridLivePreview');
    const tok = (document.getElementById('eridTokenInp')?.value || '').trim().replace(/^erid:\s*/i, '');
    if (tok) document.getElementById('wmEridStampInp').value = `Реклама • erid: ${tok}`;
    if (f) { b.style.display = 'block'; b.textContent = f; } else b.style.display = 'none';
    renderWatermarkCanvas();
  }
  function renderTgPostsHtml(posts) {
    const footer = buildErid();
    return (posts || []).map((p, i) => `
      <div class="bento-card">
        <div class="script-head"><span class="mini-tag plat">ПОСТ #0${i+1} · ${esc(p.style)}</span><button class="btn-chip accent" onclick="copyText(${JSON.stringify(`${p.headline}\n\n${p.body}\n\n${p.cta}${footer ? '\n\n' + footer : ''}`)})">📋 КОПИРОВАТЬ</button></div>
        <div style="font-weight:800; font-size:14px; margin-bottom:6px;">${esc(p.headline)}</div>
        <div style="white-space:pre-line; font-size:13px; color:#e4e4ef; margin-bottom:8px;">${esc(p.body)}</div>
        <div style="font-weight:700; color:#fff;">${esc(p.cta)}</div>
        ${footer ? `<div class="erid-tag-box">${esc(footer)}</div>` : ''}
      </div>`).join('');
  }
  async function runStandaloneTgPosts() {
    const q = document.getElementById('tgQueryInp').value.trim();
    if (!q) { toast('Вставь ссылку на товар'); return; }
    if (!S.devMode && S.balance < TG_POST_PRICE) { openCheckoutSheet(0); return; }
    const btn = document.getElementById('tgGenerateBtn'), out = document.getElementById('tgPostsListArea');
    btn.disabled = true; out.innerHTML = `<div class="bento-card mono" style="text-align:center; color:#fff;">⚡ Пишу 3 рекламных поста строго по товару...</div>`;
    try {
      const resp = await fetch('/api/tg-post', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, style: S.tone, lang: S.lang }) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Ошибка');
      if (!S.devMode) { S.balance -= TG_POST_PRICE; S.spent += TG_POST_PRICE; logTx('tg', -TG_POST_PRICE); }
      S.gens++; saveAppState(); syncHeaderAndPrices();
      if (data.meta_image) loadWatermarkImg(data.meta_image);
      out.innerHTML = renderTgPostsHtml(data.tg_posts || []); haptic('success');
    } catch (e) { out.innerHTML = `<div class="bento-card" style="color:#fda4af;">⚠️ ${esc(e.message)}</div>`; }
    finally { btn.disabled = false; }
  }

  let wmPosition = 'br', wmLoadedImg = null;
  function setWmPosition(pos, btn) {
    wmPosition = pos;
    btn.parentElement.querySelectorAll('.btn-chip').forEach(b => b.classList.remove('accent'));
    btn.classList.add('accent');
    renderWatermarkCanvas();
  }
  function loadWatermarkImg(u) {
    const img = new Image(); img.crossOrigin = 'anonymous';
    img.onload = () => { wmLoadedImg = img; renderWatermarkCanvas(); };
    img.src = '/api/proxy-image?url=' + encodeURIComponent(u);
  }
  function onCustomWatermarkFile(e) {
    const f = e.target.files?.[0]; if (!f) return;
    const r = new FileReader();
    r.onload = ev => { const img = new Image(); img.onload = () => { wmLoadedImg = img; renderWatermarkCanvas(); }; img.src = ev.target.result; };
    r.readAsDataURL(f);
  }
  function renderWatermarkCanvas() {
    const cv = document.getElementById('wmCanvas'); if (!cv) return;
    const ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
    ctx.fillStyle = '#07070a'; ctx.fillRect(0, 0, W, H);
    if (wmLoadedImg) {
      const ratio = Math.max(W / wmLoadedImg.width, H / wmLoadedImg.height);
      ctx.drawImage(wmLoadedImg, (W - wmLoadedImg.width * ratio) / 2, (H - wmLoadedImg.height * ratio) / 2, wmLoadedImg.width * ratio, wmLoadedImg.height * ratio);
    } else {
      ctx.fillStyle = '#6b6b7b'; ctx.font = '600 18px Inter, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Фото подгрузится по ссылке', W / 2, H / 2);
    }
    const h = (document.getElementById('wmHandleInp')?.value || '').trim();
    const er = (document.getElementById('wmEridStampInp')?.value || '').trim();
    if (h) {
      ctx.save();
      if (wmPosition === 'center') {
        ctx.translate(W / 2, H / 2); ctx.rotate(-Math.PI / 6);
        ctx.font = '800 38px Unbounded, sans-serif'; ctx.fillStyle = 'rgba(255,255,255,0.28)';
        ctx.textAlign = 'center'; ctx.fillText(h, 0, 0);
      } else {
        ctx.font = '700 20px Inter, sans-serif'; const tw = ctx.measureText(h).width;
        const bw = tw + 32, bh = 38;
        let bx = W - bw - 20, by = H - bh - 20;
        if (wmPosition === 'bl') bx = 20;
        if (wmPosition === 'tl') { bx = 20; by = 20; }
        ctx.fillStyle = 'rgba(5,5,8,0.82)'; ctx.beginPath(); ctx.roundRect(bx, by, bw, bh, 10); ctx.fill();
        ctx.fillStyle = '#ffffff'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(h, bx + 16, by + bh / 2);
      }
      ctx.restore();
    }
    if (er) {
      ctx.font = '600 13px JetBrains Mono, monospace'; const tw = ctx.measureText(er).width;
      const by = H - 48 - (wmPosition === 'bl' ? 48 : 0);
      ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.beginPath(); ctx.roundRect(20, by, tw + 20, 28, 7); ctx.fill();
      ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(er, 30, by + 14);
    }
  }
  function downloadWatermarkPng() {
    const cv = document.getElementById('wmCanvas'); const a = document.createElement('a');
    a.href = cv.toDataURL('image/png'); a.download = 'sniper_product.png'; a.click(); toast('Скачано');
  }

  // Cabinet with Real Telegram Avatar, Username, Stats & Transactions
  let selectedPackIdx = 1, activePayMethod = 'stars';
  function renderCabinetView() {
    const fullName = tgUser ? ([tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ') || tgUser.username || 'Creator') : 'Creator';
    const userSub = tgUser ? (tgUser.username ? '@' + tgUser.username : 'ID ' + tgUser.id) : 'Telegram Mini App';
    const initial = (fullName || 'U').trim().charAt(0).toUpperCase();
    const avatarHtml = (tgUser && tgUser.photo_url)
      ? `<div class="prof-avatar"><img src="${esc(tgUser.photo_url)}" alt="" onerror="this.parentNode.textContent='${esc(initial)}'"></div>`
      : `<div class="prof-avatar">${esc(initial)}</div>`;
    const roleBadge = S.devMode
      ? `<span class="mini-tag plat">⚡ OWNER // DEV MODE</span>`
      : `<span class="mini-tag">PRO STUDIO</span>`;

    document.getElementById('cabProfileHeader').innerHTML = `
      ${avatarHtml}
      <div style="min-width:0;">
        <div style="font-weight:800; font-size:16.5px; color:#fff;">${esc(fullName)}</div>
        <div style="display:flex; align-items:center; gap:7px; margin-top:4px; flex-wrap:wrap;">
          <span class="mono" style="font-size:12px; color:var(--muted);">${esc(userSub)}</span>
          ${roleBadge}
        </div>
      </div>
    `;

    document.getElementById('ownerDevConsole').style.display = S.devMode ? 'block' : 'none';
    document.getElementById('cabBigBalance').textContent = S.devMode ? '∞ DEV' : `${S.balance} ★`;

    document.getElementById('cabStatsGrid').innerHTML = `
      <div class="stat-box"><div class="stat-val">${S.gens}</div><div class="stat-lbl">${esc(t('stat_gens'))}</div></div>
      <div class="stat-box"><div class="stat-val">${S.spent} ★</div><div class="stat-lbl">${esc(t('stat_spent'))}</div></div>
    `;

    document.getElementById('cabPackCards').innerHTML = PACKAGES.map((p, i) => `
      <div onclick="openCheckoutSheet(${i})" style="background:#07070a; border:1.5px solid ${p.hot ? '#fff' : 'var(--border-light)'}; border-radius:14px; padding:10px 4px; text-align:center; cursor:pointer;">
        <div class="mono" style="font-size:9px; color:${p.hot ? '#fff' : 'var(--muted)'};">${esc(p.tag)}</div>
        <div style="font-family:var(--display); font-size:16px; font-weight:800; margin:2px 0;">+${p.credits} ★</div>
        <div style="font-size:11px; color:var(--muted);">${esc(p.rub)}</div>
      </div>`).join('');

    const hist = document.getElementById('cabHistoryContainer');
    hist.innerHTML = S.history.length ? S.history.map((it, idx) => `
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; padding:9px 0; border-bottom:1px solid var(--border);">
        <div style="min-width:0;">
          <div style="font-weight:700; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(it.title)}</div>
          <div class="mono" style="font-size:10.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(it.query)}</div>
        </div>
        <button class="btn-chip accent" onclick="loadHist(${idx})">Открыть</button>
      </div>`).join('') : `<div style="color:var(--muted); font-size:12px; text-align:center; padding:10px;">История пуста</div>`;

    const txEl = document.getElementById('cabTxContainer');
    txEl.innerHTML = (S.tx && S.tx.length) ? S.tx.slice(0, 12).map(x => `
      <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid var(--border);">
        <span style="font-size:12.5px;">${esc(t('tx_' + x.type))}</span>
        <strong class="mono" style="font-size:12.5px; color:${x.amount >= 0 ? '#fff' : 'var(--muted)'};">${x.amount >= 0 ? '+' : ''}${x.amount} ★</strong>
      </div>`).join('') : `<div style="color:var(--muted); font-size:12px; text-align:center; padding:10px;">Операций пока нет</div>`;
  }

  function loadHist(idx) {
    if (!S.history[idx]) return;
    lastResultData = S.history[idx].data; renderResults(lastResultData);
    switchView('studio'); document.getElementById('resultsArea').style.display = 'block';
  }
  function wipeHistory() { S.history = []; saveAppState(); renderCabinetView(); toast('История очищена'); }

  function openCheckoutSheet(pIdx) {
    if (typeof pIdx === 'number') selectedPackIdx = pIdx;
    const p = PACKAGES[selectedPackIdx] || PACKAGES[1];
    const packRow = PACKAGES.map((pk, idx) => `
      <button type="button" onclick="openCheckoutSheet(${idx})" style="flex:1; background:${idx === selectedPackIdx ? 'var(--silver-soft)' : '#07070a'}; border:1.5px solid ${idx === selectedPackIdx ? '#fff' : 'var(--border)'}; color:var(--text); border-radius:11px; padding:8px 2px; cursor:pointer;">
        <div style="font-weight:800;">+${pk.credits} ★</div><div class="mono" style="font-size:10px; color:var(--muted);">${activePayMethod === 'stars' ? pk.stars + ' ★' : (activePayMethod === 'sbp' ? pk.rub : pk.usdt)}</div>
      </button>`).join('');

    let body = '';
    if (activePayMethod === 'stars') {
      body = `<button id="starsBtn" class="btn-primary" onclick="payStars(${selectedPackIdx})">⭐ ОПЛАТИТЬ ${p.stars} STARS</button>`;
    } else if (activePayMethod === 'sbp') {
      body = `
        <div style="background:#07070a; border:1px solid var(--border); border-radius:14px; padding:12px; text-align:center; margin-bottom:10px;">
          <div style="font-weight:800; margin-bottom:6px;">СБП / Карты РФ (Демо): ${esc(p.rub)} за +${p.credits} ★</div>
          <div class="qr-mock"><svg viewBox="0 0 100 100" width="120" height="120"><rect width="30" height="30" x="5" y="5" fill="#000"/><rect width="18" height="18" x="11" y="11" fill="#fff"/><rect width="10" height="10" x="15" y="15" fill="#000"/><rect width="30" height="30" x="65" y="5" fill="#000"/><rect width="18" height="18" x="71" y="11" fill="#fff"/><rect width="10" height="10" x="75" y="15" fill="#000"/><rect width="30" height="30" x="5" y="65" fill="#000"/><rect width="18" height="18" x="11" y="71" fill="#fff"/><rect width="10" height="10" x="15" y="75" fill="#000"/></svg></div>
          <div style="font-size:11px; color:var(--muted);">Демо-шлюз: нажмите кнопку ниже для тестового зачисления</div>
        </div>
        <button class="btn-primary" onclick="demoCredit(${p.credits}, 'СБП')">⚡ ПОДТВЕРДИТЬ ОПЛАТУ (${esc(p.rub)})</button>`;
    } else {
      body = `
        <div style="background:#07070a; border:1px solid var(--border); border-radius:14px; padding:12px; margin-bottom:10px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:6px;"><span style="color:var(--muted);">К переводу (USDT TRC-20 / TON):</span><strong>${esc(p.usdt)} (+${p.credits} ★)</strong></div>
          <div class="mono" style="font-size:10.5px; background:var(--card); padding:8px; border-radius:8px; word-break:break-all;">TQn9Y2khEsLJW1ChVWFMSMeRDow5KcbLSE</div>
        </div>
        <button class="btn-primary" onclick="demoCredit(${p.credits}, 'USDT')">💎 ПРОВЕРИТЬ ТРАНЗАКЦИЮ (ДЕМО)</button>`;
    }

    openSheet(`
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;"><div style="font-weight:800; font-size:17px;">Пополнение баланса</div><button class="btn-chip" onclick="closeSheet()">✕</button></div>
      <div class="pay-methods">
        <button class="pm-btn ${activePayMethod === 'stars' ? 'active' : ''}" onclick="setPayMethod('stars')">★ Stars</button>
        <button class="pm-btn ${activePayMethod === 'sbp' ? 'active' : ''}" onclick="setPayMethod('sbp')">💳 СБП / РФ</button>
        <button class="pm-btn ${activePayMethod === 'crypto' ? 'active' : ''}" onclick="setPayMethod('crypto')">💎 USDT</button>
      </div>
      <div style="display:flex; gap:6px; margin-bottom:12px;">${packRow}</div>
      ${body}
      <div style="text-align:center; margin-top:12px;"><button style="background:none; border:none; color:var(--muted); font-size:12px; text-decoration:underline; cursor:pointer;" onclick="openSupportChat()">Вопросы по оплате? Саппорт @onadash</button></div>
    `);
  }
  function setPayMethod(m) { activePayMethod = m; openCheckoutSheet(selectedPackIdx); }
  function demoCredit(cr, l) { S.balance += cr; logTx('topup', cr); saveAppState(); syncHeaderAndPrices(); renderCabinetView(); closeSheet(); toast(`+${cr} ★ зачислено (${l})`); haptic('success'); }
  async function payStars(idx) {
    const p = PACKAGES[idx]; if (!p) return;
    const btn = document.getElementById('starsBtn'); if (btn) { btn.disabled = true; btn.textContent = 'Создаём счёт...'; }
    try {
      const resp = await fetch('/api/create-stars-invoice', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ pack_id: p.id, user_id: tgUser?.id || 0, lang: S.lang }) });
      const d = await resp.json();
      if (!resp.ok || !d.invoice_link) throw new Error(d.detail || 'Ошибка');
      if (tg?.openInvoice) {
        tg.openInvoice(d.invoice_link, s => { if (s === 'paid') demoCredit(p.credits, 'Stars'); else if (btn) { btn.disabled = false; btn.textContent = `⭐ ОПЛАТИТЬ ${p.stars} STARS`; } });
      } else window.open(d.invoice_link, '_blank');
    } catch (e) { toast(e.message || 'Ошибка'); if (btn) { btn.disabled = false; btn.textContent = `⭐ ОПЛАТИТЬ ${p.stars} STARS`; } }
  }

  function redeemPromoCode() {
    const inp = document.getElementById('promoCodeInp'), c = (inp.value || '').trim().toUpperCase();
    if (!c) return;
    if (c === 'DEVMODE' || c === 'DEVELOPER') { inp.value = ''; setDevModeState(true); return; }
    if (S.promoUsed) { toast('Промокод уже использован'); return; }
    const b = PROMOS[c]; if (!b) { toast('Неверный промокод'); return; }
    S.promoUsed = true; S.balance += b; logTx('promo', b); saveAppState(); syncHeaderAndPrices(); renderCabinetView(); inp.value = '';
    toast(`+${b} ★ зачислено!`); haptic('success');
  }

  applyTranslations();
  renderWatermarkCanvas();
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
