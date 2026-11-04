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
APP_URL_VERSIONED = APP_URL.rstrip("/") + "/?v=10"

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
    except Exception as e:
        logging.warning(f"HttpOptions fallback: {e}")
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
            logging.info(f"Discovered stable flash models: {models[:6]}")
        except Exception as e:
            logging.warning(f"Model discovery skipped: {e}")
        _MODEL_CACHE["ts"] = now
    discovered = _MODEL_CACHE["models"]
    combined = static + [m for m in discovered if m not in static]
    return list(dict.fromkeys(combined))[:5]


# =====================================================================
# SCHEMAS & PACKAGES (100 STARS / SCRIPT, 75 STARS / POST)
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
        "rub": "190 ₽",
        "usdt": "2.0 USDT",
        "title_ru": "100 ★ SNIPER UGC (1 сценарий)",
        "title_en": "100 ★ SNIPER UGC (1 Script)",
        "desc_ru": "Пополнение баланса на 100 ★ (1 полный сценарий с суфлёром или TG-пост)",
        "desc_en": "Top up 100 ★ (1 full script funnel or Telegram Ad Post)",
    },
    "pack_300": {
        "credits": 300,
        "stars": 300,
        "rub": "490 ₽",
        "usdt": "5.2 USDT",
        "title_ru": "300 ★ SNIPER UGC (Хит)",
        "title_en": "300 ★ SNIPER UGC (Popular)",
        "desc_ru": "Пакет на 3 полных сценария (по 100 ★) или 4 рекламных TG-поста (по 75 ★)",
        "desc_en": "Pack for 3 full scripts (100 ★ each) or 4 Telegram Ad Posts (75 ★ each)",
    },
    "pack_1000": {
        "credits": 1000,
        "stars": 850,
        "rub": "1 390 ₽",
        "usdt": "14.5 USDT",
        "title_ru": "1000 ★ SNIPER PRO (-15%)",
        "title_en": "1000 ★ SNIPER PRO (-15%)",
        "desc_ru": "PRO-пакет для селлеров и креаторов: 1000 ★ на баланс",
        "desc_en": "PRO pack for sellers and creators: 1000 ★ balance",
    },
}


# =====================================================================
# ERROR HANDLING
# =====================================================================
class AppError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


ERR = {
    "ru": {
        "no_key": "GEMINI_API_KEY не задан на сервере (Render → Environment).",
        "empty": "Нейросеть вернула пустой ответ. Нажми ещё раз — звёзды не списались.",
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
        "no_key": "GEMINI_API_KEY is not set on the server.",
        "empty": "Model returned an empty response. Please try again.",
        "bad_json": "Invalid response format. Please try again.",
        "busy": "AI servers are busy right now. Please retry in a few seconds.",
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
    lang = lang if lang in ERR else "ru"
    code = classify(e)
    text = ERR[lang].get(code, ERR[lang]["unknown"])
    if code == "unknown":
        text += f": {str(e)[:140]}"
    return text


# =====================================================================
# ANTI-HALLUCINATION PRODUCT PARSER (WB / OZON / WEB)
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
        "title": f"Товар Wildberries #{nm_id}",
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
                if info["title"].startswith("Товар Wildberries"):
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
        source_label = "Ozon" if "ozon." in domain else ("Яндекс Маркет" if "market.yandex" in domain else domain[:22])
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

                    # Extract strictly product JSON-LD without page recommendation noise
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
        except Exception as e:
            logging.warning(f"URL parsing warning: {e}")

        _PREVIEW_CACHE[cache_key] = fallback_data
        if len(_PREVIEW_CACHE) > 80:
            _PREVIEW_CACHE.popitem(last=False)
        return fallback_data

    return {
        "source": "Прямой запрос",
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
# STRICT RUSSIAN GEMINI GENERATION (ZERO HALLUCINATIONS)
# =====================================================================
SYSTEM_INSTRUCTION = """
Ты — элитный креативный продюсер виральных роликов (TikTok, Reels, Shorts) и рекламных интеграций в Telegram.

ЖЕЛЕЗОБЕТОННЫЕ ПРАВИЛА:
1. ЯЗЫК: Пиши АБСОЛЮТНО ВСЕ текстовые поля СТРОГО НА РУССКОМ ЯЗЫКЕ (живой, современный русский язык без машинного перевода). Единственное исключение — поля `ai_video_prompt` и `video_generator_prompts`, они пишутся на английском для видео-нейросетей.
2. ЗАПРЕТ НА ГАЛЛЮЦИНАЦИИ: Описывай ТОЛЬКО тот физический товар, который указан в `title`, `category`, `description` и `features`. НИКОГДА не придумывай характеристики от других товаров (не пиши про зарядки, кабели, аккумуляторы или электронику, если товар — это диффузор, одежда, косметика или предмет декора!).
3. ФОРМАТ ОТВЕТА: Всегда возвращай СТРОГО валидный JSON без markdown-оберток.

СХЕМА JSON:
{
  "product_title": "Точное и понятное название товара на русском",
  "category": "Категория товара на русском",
  "target_audience": "Кто целевая аудитория (1 емкое предложение на русском)",
  "pain_points": ["Боль ЦА 1", "Боль ЦА 2", "Боль ЦА 3"],
  "killer_features": ["Реальная фишка товара 1", "Реальная фишка товара 2", "Реальная фишка товара 3"],
  "viral_hooks": ["Хук 1 (первые 2 сек)", "Хук 2", "Хук 3", "Хук 4", "Хук 5"],
  "scripts": [
    {
      "title": "Название концепции на русском",
      "angle": "Угол подачи (например: Честный затест / Эстетика / Решение проблемы)",
      "virality_score": 96,
      "sound_vibe": "Под какой трек или ASMR-звук снимать (на русском)",
      "hook": "Фраза хука для первых 2 секунд на русском",
      "scenes": [
        {
          "timecode": "0:00-0:03",
          "visual": "Что показывать в кадре (на русском)",
          "voiceover": "Живой текст озвучки для чтения с суфлёра (на русском, до 18 слов)",
          "on_screen_text": "Короткая плашка текста на экране (на русском, до 6 слов)",
          "ai_video_prompt": "Cinematic b-roll prompt in English for Kling/Runway"
        }
      ],
      "cta": "Призыв к действию в конце ролика (на русском)"
    }
  ],
  "tg_posts": [
    {
      "style": "Личная рекомендация / Нативка",
      "headline": "Цепляющий заголовок поста с эмодзи на русском",
      "body": "Живой, конкретный текст поста для Telegram строго про этот товар (3-4 абзаца, с буллитами реальных плюсов товара, без воды)",
      "cta": "Призыв перейти по ссылке или артикулу"
    }
  ],
  "video_generator_prompts": ["English prompt 1", "English prompt 2"],
  "seo_caption": "Продающее описание под ролик на русском (2 предложения)",
  "hashtags": ["#хэштег1", "#хэштег2", "#хэштег3", "#хэштег4", "#хэштег5"]
}
"""

REWRITE_SYSTEM_INSTRUCTION = """
Ты — топовый сценарист коротких вертикальных роликов (Reels / TikTok / Shorts).
Твоя задача — переписать переданный сценарий строго по команде пользователя, сохранив привязку исключительно к указанному товару.
Все тексты (title, angle, sound_vibe, hook, visual, voiceover, on_screen_text, cta) пиши СТРОГО НА РУССКОМ ЯЗЫКЕ.
Верни СТРОГО валидный JSON без markdown-оберток.

СХЕМА JSON:
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
Ты — лучший копирайтер рекламных и нативных постов для Telegram-каналов и блогеров.
ПРАВИЛА:
1. Пиши СТРОГО НА РУССКОМ ЯЗЫКЕ.
2. Пиши ТОЛЬКО про конкретный переданный товар и его реальные свойства. Никаких выдуманных технических деталей от посторонних предметов!
3. Сделай 3 разных по подаче поста:
   - Пост 1: Личная нативная рекомендация от лица блогера («Девочки, это находка...» / «Заказал на тест и офигел...»).
   - Пост 2: Структурированный обзор «Находка дня» с короткими буллитами преимуществ.
   - Пост 3: Триггерный оффер через решение боли и выгоду.
Верни СТРОГО валидный JSON.

СХЕМА JSON:
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
    "shorter": "Сделай сценарий на 40% короче, динамичнее, убери всё лишнее, оставь только самые хлёсткие короткие фразы.",
    "longer": "Сделай сценарий подробнее: раскрой детали использования товара, добавь убедительных аргументов и распиши 4-5 насыщенных сцен.",
    "bolder": "Сделай хук и озвучку максимально дерзкими, триггерными и кликбейтными, чтобы остановить скролл с первой секунды.",
    "native": "Перепиши озвучку максимально естественно и разговорно, без единого рекламного штампа — как живой совет близкому другу.",
    "funny": "Добавь жизненного юмора, самоиронии и харизматичной подачи в хук и озвучку.",
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
                logging.warning(f"{model_name} bad JSON ({ae.code}), attempt {attempt + 1}")
                continue
            except Exception as e:
                msg = str(e)
                if any(k in msg for k in GONE_MARKERS):
                    gone_error = e
                    logging.warning(f"Model {model_name} unavailable, switching immediately.")
                    break
                if any(k in msg for k in TRANSIENT_MARKERS):
                    busy_error = e
                    logging.warning(f"Model {model_name} overloaded (attempt {attempt + 1}), fast fallback...")
                    await asyncio.sleep(0.8 * (attempt + 1))
                    break  # Immediately try next fallback model instead of hanging on overloaded model
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
    clean_context = {
        "источник": data.get("source", ""),
        "артикул": data.get("id", ""),
        "название_товара": data.get("title", ""),
        "бренд": data.get("brand", ""),
        "категория": data.get("category", ""),
        "цена": data.get("price", ""),
        "описание_товара": data.get("description", ""),
        "характеристики": data.get("features", []),
    }
    user_prompt = f"""
ТОЧНЫЕ ДАННЫЕ О ТОВАРЕ:
{json.dumps(clean_context, ensure_ascii=False, indent=2)}

ВЫБРАННЫЙ ФОРМАТ РОЛИКА: {FORMAT_LABELS.get(format_type, format_type)}
ВЫБРАННАЯ ТОНАЛЬНОСТЬ: {TONE_LABELS.get(tone, tone)}

ЗАДАНИЕ:
1. Напиши все тексты СТРОГО НА РУССКОМ ЯЗЫКЕ.
2. Используй ТОЛЬКО реальные свойства этого товара (если это диффузор для дома — пиши про аромат, стойкость, интерьер и уют; никаких выдумок про посторонние товары!).
3. Создай 2 детальных сценария по 4 сцены в каждом, 5 триггерных хуков, 3 боли ЦА, 3 реальные фишки товара и 3 готовых рекламных поста для Telegram (tg_posts).
"""
    parsed = await _call_gemini_json(SYSTEM_INSTRUCTION, user_prompt, require_key="scripts")
    parsed["meta_source"] = data.get("source", "Web")
    parsed["meta_image"] = data.get("image", "")
    parsed["meta_price"] = data.get("price", "")
    parsed["meta_brand"] = data.get("brand", "")
    parsed["meta_id"] = data.get("id", "")
    return parsed


async def rewrite_single_script(req: RewriteScriptRequest) -> dict:
    instruction = (
        req.custom_instruction.strip()
        if req.action == "custom" and req.custom_instruction.strip()
        else REWRITE_ACTIONS.get(req.action, REWRITE_ACTIONS["shorter"])
    )
    user_prompt = f"""
ТОВАР: {req.product_title}
ТЕКУЩИЙ СЦЕНАРИЙ (JSON):
{json.dumps(req.script, ensure_ascii=False)}

ЧТО НУЖНО СДЕЛАТЬ СО СЦЕНАРИЕМ:
{instruction}

ВАЖНО: Напиши весь текст СТРОГО НА РУССКОМ ЯЗЫКЕ (кроме ai_video_prompt). Верни объект внутри {{"script": ...}}.
"""
    parsed = await _call_gemini_json(REWRITE_SYSTEM_INSTRUCTION, user_prompt, require_key="script")
    return parsed["script"]


async def generate_tg_posts_only(data: dict, style: str, lang: str = "ru") -> dict:
    clean_context = {
        "название_товара": data.get("title", ""),
        "бренд": data.get("brand", ""),
        "категория": data.get("category", ""),
        "цена": data.get("price", ""),
        "описание_товара": data.get("description", ""),
        "характеристики": data.get("features", []),
    }
    user_prompt = f"""
ТОЧНЫЕ ДАННЫЕ О ТОВАРЕ:
{json.dumps(clean_context, ensure_ascii=False, indent=2)}

ЖЕЛАЕМАЯ ПОДАЧА: {TONE_LABELS.get(style, style)}

Напиши 3 разных рекламных поста для Telegram СТРОГО НА РУССКОМ ЯЗЫКЕ и строго про этот конкретный товар (его реальное назначение, плюсы и выгоду).
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
    key = ("ugc_ru", query.strip().lower(), format_type, tone)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    data = await extract_product_data(query)
    result = await generate_ugc_payload(data, format_type, tone, "ru")

    _CACHE[key] = result
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return result


# =====================================================================
# TELEGRAM BOT (WELCOME + QUICK PARSE + SUPPORT @onadash)
# =====================================================================
WELCOME_MSG_RU = (
    "🖤 <b>SNIPER // BLACK EDITION 6.0</b>\n"
    "<i>Профессиональная AI-студия виральных сценариев, телесуфлёра и рекламных посевов для TikTok, Reels, Shorts и Telegram</i>\n\n"
    "━━━━━━━━━━━━━━━\n"
    "⚡ <b>ЧТО ВНУТРИ СТУДИИ:</b>\n"
    "• <b>Умный парсер WB / Ozon / Сайтов:</b> вставляешь ссылку — карточка товара с фото и ценой подтягивается мгновенно.\n"
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
# FASTAPI SERVER & ENDPOINTS
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


app = FastAPI(title="SNIPER UGC BLACK EDITION", lifespan=lifespan)


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
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail=ERR["ru"]["short_query"])
    try:
        return await run_pipeline(req.query, req.format_type, req.tone, "ru")
    except Exception as e:
        logging.error(f"API analyze error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, "ru"))


@app.post("/api/rewrite-script")
async def api_rewrite_script_endpoint(req: RewriteScriptRequest):
    try:
        updated = await rewrite_single_script(req)
        return {"script": updated}
    except Exception as e:
        logging.error(f"Rewrite error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, "ru"))


@app.post("/api/tg-post")
async def api_tg_post_endpoint(req: TgPostRequest):
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail=ERR["ru"]["short_query"])
    try:
        data = await extract_product_data(req.query)
        return await generate_tg_posts_only(data, req.style, "ru")
    except Exception as e:
        logging.error(f"TG Post API error: {e}", exc_info=True)
        code = classify(e)
        status = 503 if code in ("busy", "quota", "timeout") else 500
        raise HTTPException(status_code=status, detail=error_text(e, "ru"))


@app.post("/api/create-stars-invoice")
async def api_create_stars_invoice(req: StarsInvoiceRequest):
    pack = STAR_PACKS.get(req.pack_id)
    if not pack:
        raise HTTPException(status_code=400, detail="Unknown package")
    if not bot:
        raise HTTPException(status_code=503, detail="BOT_TOKEN is not configured")
    try:
        invoice_link = await bot.create_invoice_link(
            title=pack["title_ru"],
            description=pack["desc_ru"],
            payload=f"{req.pack_id}:{req.user_id}:{pack['credits']}",
            provider_token="",
            currency="XTR",
            prices=[LabeledPrice(label=pack["title_ru"], amount=pack["stars"])],
        )
        return {"invoice_link": invoice_link, "credits": pack["credits"], "stars": pack["stars"]}
    except Exception as e:
        logging.error(f"Stars invoice error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stars invoice error: {e}")


# =====================================================================
# BESPOKE OLED BLACK EDITION FRONTEND
# =====================================================================
FRONTEND_UI = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<title>SNIPER // BLACK EDITION</title>
<script src="[https://telegram.org/js/telegram-web-app.js](https://telegram.org/js/telegram-web-app.js)"></script>
<link rel="preconnect" href="[https://fonts.googleapis.com](https://fonts.googleapis.com)">
<link rel="preconnect" href="[https://fonts.gstatic.com](https://fonts.gstatic.com)" crossorigin>
<link href="[https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Syne:wght@700;800&family=JetBrains+Mono:wght@500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Syne:wght@700;800&family=JetBrains+Mono:wght@500;600;700&display=swap)" rel="stylesheet">
<style>
  :root {
    --bg: #040406;
    --card: #0b0b10;
    --card-elevated: #111118;
    --card-glass: rgba(13, 13, 19, 0.84);
    --border: rgba(255, 255, 255, 0.07);
    --border-light: rgba(255, 255, 255, 0.14);
    --border-neon: rgba(204, 255, 0, 0.45);
    --text: #f4f4f6;
    --muted: #7e7e91;
    --neon: #ccff00;
    --neon-soft: rgba(204, 255, 0, 0.12);
    --neon-glow: rgba(204, 255, 0, 0.24);
    --emerald: #10b981;
    --danger: #f43f5e;
    --radius-xl: 22px;
    --radius-lg: 16px;
    --radius-md: 12px;
    --font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --display: 'Syne', sans-serif;
    --mono: 'JetBrains Mono', monospace;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.5;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
  }

  body {
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse 80% 40% at 50% -15%, rgba(204, 255, 0, 0.08), transparent),
      radial-gradient(circle at 100% 25%, rgba(255, 255, 255, 0.025), transparent 45%);
  }

  .app {
    max-width: 500px;
    margin: 0 auto;
    padding: 12px 16px calc(110px + env(safe-area-inset-bottom));
  }
  .mono { font-family: var(--mono); }
  button { font-family: inherit; user-select: none; }

  /* ================= TOP NAVIGATION BAR ================= */
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0 18px;
  }
  .brand-cluster {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
  }
  .brand-mark {
    width: 30px;
    height: 30px;
    border-radius: 9px;
    background: var(--card-elevated);
    border: 1px solid var(--border-light);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: var(--mono);
    font-weight: 800;
    font-size: 12px;
    color: var(--neon);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
  }
  .brand-name {
    font-family: var(--display);
    font-weight: 800;
    font-size: 15px;
    letter-spacing: 0.04em;
  }
  .dev-pill {
    display: none;
    align-items: center;
    gap: 4px;
    font-family: var(--mono);
    font-size: 9.5px;
    font-weight: 700;
    color: #000;
    background: var(--neon);
    padding: 3px 8px;
    border-radius: 6px;
    box-shadow: 0 0 16px var(--neon-glow);
  }
  .dev-pill.active { display: inline-flex; }

  .wallet-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    background: var(--card-elevated);
    border: 1px solid var(--border-light);
    padding: 7px 13px;
    border-radius: 999px;
    color: var(--text);
    font-weight: 700;
    font-size: 13px;
    cursor: pointer;
    transition: transform 0.15s ease, border-color 0.2s ease;
  }
  .wallet-pill:active { transform: scale(0.95); border-color: var(--neon); }
  .wallet-star { color: var(--neon); font-size: 14px; }

  /* ================= VIEWS & ANIMATIONS ================= */
  .view { display: none; }
  .view.active {
    display: block;
    animation: fadeSlide 0.28s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  /* ================= BENTO SURFACES ================= */
  .bento-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top-color: var(--border-light);
    border-radius: var(--radius-xl);
    padding: 18px;
    margin-bottom: 14px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.55);
  }
  .section-kicker {
    font-family: var(--mono);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--neon);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .hero-heading {
    font-weight: 800;
    font-size: 21px;
    line-height: 1.22;
    letter-spacing: -0.025em;
    margin-bottom: 6px;
  }
  .hero-sub {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
  }

  /* ================= COMMAND DOCK (LINK INPUT) ================= */
  .cmd-dock {
    background: #07070b;
    border: 1.5px solid var(--border-light);
    border-radius: 18px;
    padding: 12px;
    margin: 14px 0 12px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
  }
  .cmd-dock:focus-within {
    border-color: var(--neon);
    box-shadow: 0 0 0 4px var(--neon-soft);
  }
  .cmd-top {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }
  .cmd-input {
    flex: 1;
    min-width: 0;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: inherit;
    font-size: 14.5px;
    font-weight: 500;
    line-height: 1.4;
    resize: none;
    min-height: 44px;
    max-height: 96px;
  }
  .cmd-input::placeholder { color: #525263; font-weight: 400; }

  .cmd-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    gap: 8px;
  }
  .dock-btn {
    background: var(--card-elevated);
    color: var(--text);
    border: 1px solid var(--border-light);
    padding: 7px 12px;
    border-radius: 10px;
    font-size: 11.5px;
    font-weight: 700;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }
  .dock-btn:active { transform: scale(0.95); background: var(--neon); color: #000; }

  /* SETTINGS TRIGGER PILL (REPLACES OLD CLUTTERED CHIPS) */
  .params-trigger {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-light);
    color: var(--text);
    padding: 7px 12px;
    border-radius: 10px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 7px;
    transition: border-color 0.2s ease;
  }
  .params-trigger:active { border-color: var(--neon); }
  .params-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--neon); }

  /* ================= LIVE PRODUCT OVERVIEW CARD ================= */
  .overview-card {
    display: none;
    background: linear-gradient(135deg, rgba(204, 255, 0, 0.08), rgba(255, 255, 255, 0.02)), var(--card-elevated);
    border: 1px solid var(--border-neon);
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 14px;
    animation: fadeSlide 0.25s ease;
  }
  .ov-flex { display: flex; align-items: center; gap: 12px; }
  .ov-img {
    width: 58px;
    height: 58px;
    border-radius: 12px;
    object-fit: cover;
    background: #050508;
    border: 1px solid var(--border-light);
    flex-shrink: 0;
  }
  .ov-info { flex: 1; min-width: 0; }
  .ov-badges { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; flex-wrap: wrap; }
  .ov-source {
    font-family: var(--mono);
    font-size: 9.5px;
    font-weight: 700;
    background: var(--neon);
    color: #000;
    padding: 2px 7px;
    border-radius: 5px;
    text-transform: uppercase;
  }
  .ov-price {
    font-family: var(--mono);
    font-size: 10.5px;
    font-weight: 700;
    background: rgba(16, 185, 129, 0.16);
    color: var(--emerald);
    padding: 2px 7px;
    border-radius: 5px;
  }
  .ov-title {
    font-weight: 700;
    font-size: 13.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ov-meta {
    font-size: 11.5px;
    color: var(--muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ================= PRIMARY CTA BUTTON ================= */
  .btn-primary {
    position: relative;
    overflow: hidden;
    width: 100%;
    background: var(--neon);
    color: #050507;
    border: none;
    padding: 15px 18px;
    border-radius: 15px;
    font-size: 13.5px;
    font-weight: 800;
    letter-spacing: 0.02em;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 8px 28px var(--neon-glow);
    transition: transform 0.12s ease, opacity 0.15s ease;
  }
  .btn-primary:active { transform: scale(0.98); }
  .btn-primary:disabled { opacity: 0.45; cursor: not-allowed; box-shadow: none; }

  .btn-secondary {
    width: 100%;
    background: var(--card-elevated);
    color: var(--text);
    border: 1px solid var(--border-light);
    padding: 13px 16px;
    border-radius: 14px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.15s ease;
  }
  .btn-secondary:active { transform: scale(0.98); border-color: var(--neon); }

  .sub-note {
    text-align: center;
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--muted);
    margin-top: 10px;
  }

  /* ================= AI NEURAL STREAMING CONSOLE ================= */
  .neural-box {
    display: none;
    background: #07070b;
    border: 1px solid var(--border-neon);
    border-radius: var(--radius-xl);
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.65);
  }
  .nb-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .nb-title {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--neon);
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .nb-pulse {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--neon);
    animation: ping 1s infinite alternate;
  }
  @keyframes ping { from { opacity: 0.3; transform: scale(0.8); } to { opacity: 1; transform: scale(1.2); } }
  .nb-bar-track {
    width: 100%;
    height: 5px;
    background: var(--card-elevated);
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  .nb-bar-fill {
    height: 100%;
    width: 15%;
    background: linear-gradient(90deg, var(--neon), var(--emerald));
    transition: width 0.4s ease;
  }
  .nb-log {
    font-family: var(--mono);
    font-size: 11.5px;
    line-height: 1.6;
    color: #d4d4e0;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    min-height: 110px;
  }
  .nb-line { margin-bottom: 5px; display: flex; gap: 8px; }
  .nb-line .arrow { color: var(--neon); }

  /* ================= RESULTS & SCRIPTS ================= */
  #resultsArea { display: none; }
  .seg-tabs {
    display: flex;
    gap: 4px;
    overflow-x: auto;
    scrollbar-width: none;
    background: var(--card);
    padding: 5px;
    border: 1px solid var(--border-light);
    border-radius: 14px;
    margin-bottom: 14px;
  }
  .seg-tabs::-webkit-scrollbar { display: none; }
  .seg-btn {
    flex: 1;
    white-space: nowrap;
    background: transparent;
    color: var(--muted);
    border: none;
    padding: 8px 10px;
    font-size: 11px;
    font-weight: 700;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .seg-btn.active { background: var(--text); color: #050507; }
  .seg-pane { display: none; }
  .seg-pane.active { display: block; animation: fadeSlide 0.22s ease; }

  .script-head {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    padding-bottom: 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .script-badge-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 4px;
  }
  .mini-tag {
    font-family: var(--mono);
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 6px;
    background: var(--card-elevated);
    border: 1px solid var(--border-light);
    color: var(--muted);
  }
  .mini-tag.neon {
    color: var(--neon);
    border-color: var(--border-neon);
    background: var(--neon-soft);
  }

  .btn-chip {
    background: var(--card-elevated);
    border: 1px solid var(--border-light);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 9px;
    font-family: var(--mono);
    font-size: 10.5px;
    font-weight: 700;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .btn-chip:active { background: var(--neon); color: #000; }
  .btn-chip.accent {
    background: var(--neon);
    color: #000;
    border-color: var(--neon);
  }

  .hook-callout {
    background: #07070b;
    border-left: 3px solid var(--neon);
    padding: 11px 13px;
    border-radius: 0 12px 12px 0;
    margin-bottom: 10px;
    font-size: 13.5px;
    font-weight: 600;
  }
  .scene-block {
    background: #07070b;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 12px;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .scene-time {
    font-family: var(--mono);
    font-size: 10.5px;
    color: var(--neon);
    margin-bottom: 4px;
  }
  .prompt-strip {
    margin-top: 8px;
    padding: 8px 10px;
    background: var(--card);
    border: 1px dashed var(--border-light);
    border-radius: 9px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }

  /* AI REWRITE BAR */
  .rw-box {
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px dashed var(--border-light);
  }
  .rw-title {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }
  .rw-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .rw-pill {
    background: #07070b;
    border: 1px solid var(--border-light);
    color: var(--text);
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .rw-pill:active { background: var(--neon); color: #000; border-color: var(--neon); }

  /* ================= DEDICATED TELEPROMPTER TAB ================= */
  .tp-script-pick {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: #07070b;
    border: 1px solid var(--border);
    border-radius: 13px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 0.15s ease;
  }
  .tp-script-pick:active, .tp-script-pick.selected {
    border-color: var(--neon);
    background: var(--neon-soft);
  }
  .tp-editor {
    width: 100%;
    min-height: 150px;
    background: #07070b;
    border: 1px solid var(--border-light);
    border-radius: 14px;
    color: var(--text);
    padding: 14px;
    font-family: inherit;
    font-size: 15px;
    line-height: 1.55;
    resize: vertical;
    outline: none;
    margin-bottom: 12px;
  }
  .tp-editor:focus { border-color: var(--neon); }

  /* FULLSCREEN TELEPROMPTER PLAYER */
  .tp-player {
    position: fixed;
    inset: 0;
    z-index: 600;
    background: #000000;
    display: none;
    flex-direction: column;
    justify-content: space-between;
    padding: env(safe-area-inset-top) 0 env(safe-area-inset-bottom);
  }
  .tp-player.open { display: flex; }
  .tp-p-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: rgba(11, 11, 16, 0.92);
    border-bottom: 1px solid var(--border);
    z-index: 5;
  }
  .tp-p-stage {
    flex: 1;
    overflow-y: auto;
    padding: 36vh 22px 45vh;
  }
  .tp-p-stage.mirror { transform: scaleX(-1); }
  .tp-eye-band {
    position: fixed;
    top: 35%;
    left: 0;
    right: 0;
    height: 92px;
    pointer-events: none;
    z-index: 3;
    border-top: 1.5px dashed var(--border-neon);
    border-bottom: 1.5px dashed var(--border-neon);
    background: linear-gradient(90deg, rgba(204,255,0,0.06), transparent 15%, transparent 85%, rgba(204,255,0,0.06));
  }
  .tp-p-text {
    font-size: 28px;
    font-weight: 800;
    line-height: 1.45;
    color: #fff;
    text-align: center;
    white-space: pre-wrap;
    outline: none;
  }
  .tp-p-controls {
    background: rgba(11, 11, 16, 0.96);
    border-top: 1px solid var(--border);
    padding: 12px 16px 16px;
    z-index: 5;
  }
  .tp-cd-overlay {
    position: fixed;
    inset: 0;
    z-index: 10;
    background: rgba(0,0,0,0.85);
    display: none;
    align-items: center;
    justify-content: center;
    font-family: var(--display);
    font-size: 84px;
    font-weight: 800;
    color: var(--neon);
  }

  /* ================= INPUTS & WATERMARK ================= */
  .field-inp {
    width: 100%;
    background: #07070b;
    border: 1px solid var(--border-light);
    border-radius: 12px;
    color: var(--text);
    padding: 11px 13px;
    font-size: 13.5px;
    font-family: inherit;
    outline: none;
  }
  .field-inp:focus { border-color: var(--neon); }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
  .erid-tag-box {
    background: rgba(204, 255, 0, 0.06);
    border: 1px dashed var(--border-neon);
    border-radius: 11px;
    padding: 9px 12px;
    font-family: var(--mono);
    font-size: 11.5px;
    color: var(--neon);
    margin-top: 8px;
  }

  /* ================= BOTTOM SHEET MODAL (SETTINGS & MULTI-CHECKOUT) ================= */
  .sheet-backdrop {
    position: fixed;
    inset: 0;
    z-index: 400;
    background: rgba(0, 0, 0, 0.76);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    display: none;
    align-items: flex-end;
    justify-content: center;
  }
  .sheet-backdrop.open { display: flex; }
  .sheet-box {
    width: 100%;
    max-width: 500px;
    max-height: 88vh;
    overflow-y: auto;
    background: var(--card-elevated);
    border: 1px solid var(--border-light);
    border-radius: 26px 26px 0 0;
    padding: 18px 18px calc(24px + env(safe-area-inset-bottom));
    animation: sheetUp 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes sheetUp { from { transform: translateY(40px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
  .sheet-handle {
    width: 38px;
    height: 4px;
    background: rgba(255,255,255,0.2);
    border-radius: 999px;
    margin: 0 auto 14px;
  }

  /* VISUAL OPTION CARDS IN SETTINGS SHEET */
  .opt-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 16px;
  }
  .opt-card {
    background: #07070b;
    border: 1.5px solid var(--border);
    border-radius: 14px;
    padding: 11px 12px;
    text-align: left;
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .opt-card.active {
    border-color: var(--neon);
    background: var(--neon-soft);
  }
  .opt-card .o-title { font-weight: 700; font-size: 13px; margin-bottom: 2px; display: flex; justify-content: space-between; }
  .opt-card .o-desc { font-size: 11px; color: var(--muted); line-height: 1.3; }

  /* PAYMENT METHOD SWITCHER IN CHECKOUT SHEET */
  .pay-methods {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    background: #07070b;
    padding: 5px;
    border: 1px solid var(--border-light);
    border-radius: 13px;
    margin-bottom: 14px;
  }
  .pm-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    padding: 9px 4px;
    border-radius: 9px;
    font-size: 11.5px;
    font-weight: 700;
    cursor: pointer;
  }
  .pm-btn.active { background: var(--neon); color: #000; }

  .qr-mock {
    width: 148px;
    height: 148px;
    margin: 10px auto;
    background: #fff;
    border-radius: 14px;
    padding: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* ================= FLOATING 4-TAB BOTTOM DOCK ================= */
  .bottom-dock {
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 100;
    padding: 8px 12px calc(10px + env(safe-area-inset-bottom));
    background: linear-gradient(180deg, rgba(4,4,6,0) 0%, rgba(4,4,6,0.94) 40%);
    pointer-events: none;
  }
  .dock-inner {
    pointer-events: auto;
    max-width: 480px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    background: var(--card-glass);
    backdrop-filter: blur(22px);
    -webkit-backdrop-filter: blur(22px);
    border: 1px solid var(--border-light);
    border-radius: 20px;
    padding: 5px;
    box-shadow: 0 14px 34px rgba(0, 0, 0, 0.75);
  }
  .dock-tab {
    background: transparent;
    border: none;
    color: var(--muted);
    padding: 7px 2px;
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    font-size: 10px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.18s ease;
  }
  .dock-tab.active {
    background: var(--neon);
    color: #050507;
    box-shadow: 0 4px 14px var(--neon-glow);
  }
  .dock-tab svg {
    width: 19px;
    height: 19px;
    stroke: currentColor;
    fill: none;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  .toast {
    position: fixed;
    bottom: calc(92px + env(safe-area-inset-bottom));
    left: 50%;
    transform: translateX(-50%);
    background: var(--neon);
    color: #000;
    padding: 9px 18px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 800;
    display: none;
    z-index: 800;
    white-space: nowrap;
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
  }
</style>
</head>
<body>
<div class="app">

  <!-- HEADER -->
  <header class="header">
    <div class="brand-cluster" onclick="onSecretLogoTap()">
      <div class="brand-mark">SU</div>
      <div>
        <div class="brand-name">SNIPER</div>
      </div>
      <span id="devBadgeTop" class="dev-pill">⚡ DEV MODE</span>
    </div>
    <button class="wallet-pill" onclick="switchView('cabinet')">
      <span class="wallet-star">★</span>
      <span id="topBalanceVal">100</span>
    </button>
  </header>

  <!-- ==================== VIEW 1: STUDIO (CLEAN WORKSPACE) ==================== -->
  <section id="view-studio" class="view active">
    <div class="bento-card">
      <div class="section-kicker"><span>●</span><span>AI CONTENT ENGINE // РУССКАЯ ВЕРСИЯ</span></div>
      <h1 class="hero-heading">Виральные сценарии за 10 секунд</h1>
      <p class="hero-sub">Вставь ссылку с Wildberries, Ozon или любого сайта — карточка товара определится автоматически.</p>

      <!-- SLEEK COMMAND DOCK -->
      <div class="cmd-dock">
        <div class="cmd-top">
          <textarea id="mainQueryInp" class="cmd-input" rows="2" placeholder="Ссылка на товар WB / Ozon, артикул или описание..." oninput="onProductInput(this, 'main')"></textarea>
        </div>
        <div class="cmd-bar">
          <div style="display:flex; gap:6px;">
            <button type="button" class="dock-btn" onclick="pasteFromClipboard('mainQueryInp', 'main')">📋 Вставить</button>
            <button type="button" id="clearMainBtn" class="dock-btn" style="display:none;" onclick="clearProductInput('mainQueryInp', 'main')">✕</button>
          </div>
          <!-- OPENS MODERN BOTTOM SHEET INSTEAD OF CLUTTERED CHIPS -->
          <button type="button" class="params-trigger" onclick="openSettingsSheet()">
            <span class="params-dot"></span>
            <span id="activeParamsSummary">UGC Обзор · Нативный</span>
            <span>⚙️</span>
          </button>
        </div>
      </div>

      <!-- AUTO LIVE PRODUCT OVERVIEW -->
      <div id="overviewMain" class="overview-card"></div>

      <button id="generateMainBtn" class="btn-primary" onclick="runFullAnalysis()">
        <span>⚡ СОЗДАТЬ СЦЕНАРИИ</span>
        <span id="mainBtnPriceTag" class="mono">(100 ★)</span>
      </button>
      <div class="sub-note" id="studioCostHint">Звёзды списываются только после успешной генерации · Рерайты бесплатно</div>
    </div>

    <div id="studioErrorBox" class="bento-card" style="display:none; border-color:var(--danger); color:#fda4af; font-size:13px;"></div>

    <!-- AI NEURAL STREAMING CONSOLE -->
    <div id="neuralBox" class="neural-box">
      <div class="nb-top">
        <div class="nb-title"><span class="nb-pulse"></span><span id="nbStatusTxt">AI ПРОДЮСЕР АКТИВЕН</span></div>
        <span class="mono" style="font-size:11px; color:var(--neon);" id="nbPctTxt">15%</span>
      </div>
      <div class="nb-bar-track"><div class="nb-bar-fill" id="nbBarFill"></div></div>
      <div class="nb-log" id="nbLogStream"></div>
    </div>

    <!-- GENERATED RESULTS -->
    <div id="resultsArea">
      <div id="resultProductCard" class="bento-card" style="padding:14px;"></div>

      <div class="seg-tabs">
        <button class="seg-btn active" onclick="switchResultTab('scripts', this)">Сценарии</button>
        <button class="seg-btn" onclick="switchResultTab('tgposts', this)">TG Посты</button>
        <button class="seg-btn" onclick="switchResultTab('hooks', this)">Топ-Хуки</button>
        <button class="seg-btn" onclick="switchResultTab('broll', this)">AI Видео</button>
        <button class="seg-btn" onclick="switchResultTab('seo', this)">ЦА & SEO</button>
      </div>

      <div id="pane-scripts" class="seg-pane active"></div>
      <div id="pane-tgposts" class="seg-pane"></div>
      <div id="pane-hooks" class="seg-pane"></div>
      <div id="pane-broll" class="seg-pane"></div>
      <div id="pane-seo" class="seg-pane"></div>
    </div>
  </section>

  <!-- ==================== VIEW 2: DEDICATED TELEPROMPTER STUDIO ==================== -->
  <section id="view-prompter" class="view">
    <div class="bento-card">
      <div class="section-kicker"><span>📟</span><span>TELEPROMPTER STUDIO</span></div>
      <h1 class="hero-heading">Студийный телесуфлёр</h1>
      <p class="hero-sub" style="margin-bottom:14px;">Выбери готовый сценарий из истории своих генераций ниже или отредактируй текст перед записью ролика.</p>

      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">Твои сгенерированные сценарии:</div>
      <div id="prompterHistoryList" style="margin-bottom:14px;"></div>

      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:6px;">Текст для чтения с камеры:</div>
      <textarea id="prompterEditorInp" class="tp-editor" placeholder="Выбери сценарий выше или вставь сюда любой свой текст для записи видео..."></textarea>

      <div class="grid-2" style="margin-bottom:14px;">
        <div style="background:#07070b; border:1px solid var(--border); border-radius:12px; padding:10px 12px;">
          <div class="mono" style="font-size:10px; color:var(--muted);">СКОРОСТЬ СКРОЛЛА: <strong style="color:var(--neon);" id="tpTabSpeedVal">1.2x</strong></div>
          <input type="range" style="width:100%; accent-color:var(--neon); margin-top:6px;" min="0.4" max="3.0" step="0.1" value="1.2" oninput="syncTpSpeed(this.value)">
        </div>
        <div style="background:#07070b; border:1px solid var(--border); border-radius:12px; padding:10px 12px;">
          <div class="mono" style="font-size:10px; color:var(--muted);">РАЗМЕР ШРИФТА: <strong style="color:var(--neon);" id="tpTabFontVal">28px</strong></div>
          <input type="range" style="width:100%; accent-color:var(--neon); margin-top:6px;" min="20" max="44" step="2" value="28" oninput="syncTpFont(this.value)">
        </div>
      </div>

      <button class="btn-primary" onclick="launchFullscreenPrompterFromTab()">
        <span>🔴 ЗАПУСТИТЬ ПОЛНОЭКРАННЫЙ СУФЛЁР</span>
      </button>
    </div>
  </section>

  <!-- ==================== VIEW 3: TG AD POSTS + ERID + WATERMARK ==================== -->
  <section id="view-tgads" class="view">
    <div class="bento-card">
      <div class="section-kicker"><span>📢</span><span>TELEGRAM AD SEEDING & ОРД</span></div>
      <h1 class="hero-heading">Рекламные посты и маркировка</h1>
      <p class="hero-sub">Создаёт 3 живых поста под конкретный товар (без воды и чужих характеристик) + автоматически собирает токен ERID.</p>

      <div class="cmd-dock">
        <textarea id="tgQueryInp" class="cmd-input" rows="2" placeholder="Вставь ссылку на рекламируемый товар WB / Ozon / Сайт..." oninput="onProductInput(this, 'tg')"></textarea>
        <div class="cmd-bar">
          <button type="button" class="dock-btn" onclick="pasteFromClipboard('tgQueryInp', 'tg')">📋 Вставить ссылку</button>
          <button type="button" id="clearTgBtn" class="dock-btn" style="display:none;" onclick="clearProductInput('tgQueryInp', 'tg')">✕ Очистить</button>
        </div>
      </div>

      <div id="overviewTg" class="overview-card"></div>

      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin:10px 0 6px; display:flex; justify-content:space-between;">
        <span>Окно маркировки рекламы (ERID / ОРД)</span>
        <span style="color:var(--neon);">ФЗ-38 РФ</span>
      </div>
      <div class="grid-2">
        <input id="eridOrgInp" class="field-inp" placeholder="Рекламодатель (ИП / ООО + ИНН)" oninput="refreshEridFooters()">
        <input id="eridTokenInp" class="field-inp mono" placeholder="erid: 2Vtzqv..." oninput="refreshEridFooters()">
      </div>
      <input id="eridUrlInp" class="field-inp" placeholder="Партнёрская ссылка или артикул для кнопки/текста" oninput="refreshEridFooters()" style="margin-bottom:10px;">
      <div id="eridLivePreview" class="erid-tag-box" style="display:none;"></div>

      <button id="tgGenerateBtn" class="btn-primary" style="margin-top:12px;" onclick="runStandaloneTgPosts()">
        <span>✨ СГЕНЕРИРОВАТЬ 3 ПОСТА</span>
        <span id="tgBtnPriceTag" class="mono">(75 ★)</span>
      </button>
    </div>

    <div id="tgPostsListArea"></div>

    <!-- WATERMARK GENERATOR CARD -->
    <div class="bento-card">
      <div class="section-kicker"><span>🖼</span><span>WATERMARK & ERID STAMP</span></div>
      <h2 style="font-size:16px; font-weight:800; margin-bottom:4px;">Водяной знак на фото товара</h2>
      <p class="hero-sub" style="margin-bottom:12px;">Фото товара подгружается автоматически из ссылки (или загрузи своё) — наложи ник канала и плашку erid.</p>

      <div class="grid-2">
        <input id="wmHandleInp" class="field-inp" placeholder="@ваш_канал" value="@sniper_ugc" oninput="renderWatermarkCanvas()">
        <input id="wmEridStampInp" class="field-inp mono" placeholder="Реклама • erid: ..." oninput="renderWatermarkCanvas()">
      </div>

      <div style="display:flex; gap:6px; margin:8px 0 10px; overflow-x:auto;">
        <button class="btn-chip accent" onclick="setWmPosition('br', this)">Справа внизу</button>
        <button class="btn-chip" onclick="setWmPosition('bl', this)">Слева внизу</button>
        <button class="btn-chip" onclick="setWmPosition('center', this)">По диагонали</button>
        <button class="btn-chip" onclick="setWmPosition('tl', this)">Сверху</button>
      </div>

      <div style="background:#050508; border:1px solid var(--border-light); border-radius:14px; overflow:hidden; margin-bottom:12px; display:flex; justify-content:center;">
        <canvas id="wmCanvas" width="600" height="750" style="max-width:100%; max-height:330px; display:block;"></canvas>
      </div>

      <div class="grid-2">
        <label class="btn-secondary" style="cursor:pointer;">
          <span>📷 Загрузить фото</span>
          <input type="file" accept="image/*" style="display:none;" onchange="onCustomWatermarkFile(event)">
        </label>
        <button class="btn-primary" onclick="downloadWatermarkPng()">
          <span>⬇ Скачать PNG</span>
        </button>
      </div>
    </div>
  </section>

  <!-- ==================== VIEW 4: CABINET, MULTI-PAY & SUPPORT ==================== -->
  <section id="view-cabinet" class="view">
    <div class="bento-card">
      <div id="cabProfileHeader" style="display:flex; align-items:center; gap:12px;"></div>
    </div>

    <!-- OWNER DEVELOPER CONSOLE -->
    <div id="ownerDevConsole" class="bento-card" style="display:none; border-color:var(--neon); background:linear-gradient(135deg, rgba(204,255,0,0.12), rgba(11,11,16,0.95));">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span class="mono" style="font-size:11.5px; font-weight:800; color:var(--neon);">⚡ DEVELOPER MODE // ROOT ACCESS</span>
        <span class="mini-tag neon">OWNER</span>
      </div>
      <p style="font-size:12.5px; color:#d4d4e0; margin-bottom:12px;">
        Персональный режим разработчика активен. Все генерации сценариев (100 ★) и постов (75 ★) работают без лимитов и списаний.
      </p>
      <div class="grid-2">
        <button class="btn-primary" style="padding:10px;" onclick="devGrantStars(1000)">+1000 ★ ТЕСТ</button>
        <button class="btn-secondary" style="padding:10px;" onclick="setDevModeState(false)">Выключить Dev</button>
      </div>
    </div>

    <!-- BALANCE & MULTI-PAYMENT TRIGGER -->
    <div class="bento-card" style="background:linear-gradient(135deg, rgba(204,255,0,0.14), rgba(17,17,24,0.95)); border-color:var(--border-neon);">
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase;">ТВОЙ БАЛАНС ЗВЁЗД</div>
      <div style="font-family:var(--display); font-size:38px; font-weight:800; color:var(--neon); line-height:1.1; margin:6px 0 4px;" id="cabBigBalance">100 ★</div>
      <div style="font-size:12px; color:var(--muted); margin-bottom:14px;">1 сценарий = 100 ★ · 1 пакет TG-постов = 75 ★</div>

      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:8px; margin-bottom:12px;" id="cabPackCards"></div>

      <button class="btn-primary" onclick="openCheckoutSheet(1)">
        <span>💳 ПОПОЛНИТЬ (STARS / СБП / USDT)</span>
      </button>
    </div>

    <!-- SUPPORT @onadash & REFERRAL -->
    <div class="grid-2" style="margin-bottom:14px;">
      <button class="btn-secondary" onclick="openSupportChat()">
        <span>💬 Саппорт @onadash</span>
      </button>
      <button class="btn-secondary" onclick="shareBotReferral()">
        <span>🎁 Другу (+50 ★)</span>
      </button>
    </div>

    <!-- PROMO CODE -->
    <div class="bento-card">
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">Активация промокода</div>
      <div style="display:flex; gap:8px;">
        <input id="promoCodeInp" class="field-inp mono" placeholder="Промокод (напр. SNIPER)" autocapitalize="characters">
        <button class="btn-primary" style="width:auto; padding:0 18px;" onclick="redeemPromoCode()">ОК</button>
      </div>
    </div>

    <!-- HISTORY -->
    <div class="bento-card">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <span style="font-weight:800; font-size:14px;">История генераций</span>
        <button style="background:none; border:none; color:var(--muted); font-size:11.5px; text-decoration:underline; cursor:pointer;" onclick="wipeHistory()">Очистить</button>
      </div>
      <div id="cabHistoryContainer"></div>
    </div>
  </section>

</div>

<!-- ==================== FULLSCREEN TELEPROMPTER PLAYER ==================== -->
<div id="tpFullscreenPlayer" class="tp-player">
  <div class="tp-p-top">
    <div class="mono" style="font-size:11.5px; color:var(--neon); font-weight:700;">🔴 REC // TELEPROMPTER</div>
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
      <input type="range" id="tpModalSpeedRange" style="flex:1; accent-color:var(--neon);" min="0.4" max="3.0" step="0.1" value="1.2" oninput="syncTpSpeed(this.value)">
      <button class="btn-chip" onclick="nudgePrompterFont(-2)">A-</button>
      <button class="btn-chip" onclick="nudgePrompterFont(2)">A+</button>
    </div>
    <button id="tpPlayActionBtn" class="btn-primary" onclick="startPrompterWithCountdown()">▶ СТАРТ С ОТСЧЁТОМ (3 СЕК)</button>
  </div>
</div>

<!-- ==================== BOTTOM SHEET (SETTINGS & MULTI-PAY CHECKOUT) ==================== -->
<div id="sheetBackdrop" class="sheet-backdrop" onclick="if(event.target===this)closeSheet()">
  <div id="sheetContent" class="sheet-box"></div>
</div>

<!-- ==================== 4-TAB FLOATING NAVIGATION DOCK ==================== -->
<nav class="bottom-dock">
  <div class="dock-inner">
    <button id="tabBtn-studio" class="dock-tab active" onclick="switchView('studio')">
      <svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z"/></svg>
      <span>Сценарии</span>
    </button>
    <button id="tabBtn-prompter" class="dock-tab" onclick="switchView('prompter')">
      <svg viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="14" rx="2"/><path d="M8 21h8"/><path d="M12 18v3"/></svg>
      <span>Суфлёр</span>
    </button>
    <button id="tabBtn-tgads" class="dock-tab" onclick="switchView('tgads')">
      <svg viewBox="0 0 24 24"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>
      <span>TG Посты</span>
    </button>
    <button id="tabBtn-cabinet" class="dock-tab" onclick="switchView('cabinet')">
      <svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-8 8-8s8 3.6 8 8"/></svg>
      <span>Кабинет</span>
    </button>
  </div>
</nav>

<div id="toastBanner" class="toast"></div>

<script>
  // ================= Telegram Init =================
  const SERVER_ADMIN_ID = "__ADMIN_TG_ID__";
  const SUPPORT_USER = "onadash";
  const tg = (window.Telegram && window.Telegram.WebApp) ? window.Telegram.WebApp : null;
  if (tg) {
    try {
      tg.ready();
      tg.expand();
      if (tg.setHeaderColor) tg.setHeaderColor('#040406');
      if (tg.setBackgroundColor) tg.setBackgroundColor('#040406');
    } catch (e) {}
  }
  const tgUser = (tg && tg.initDataUnsafe && tg.initDataUnsafe.user) ? tg.initDataUnsafe.user : null;

  function isOwnerAccount() {
    if (!tgUser) return false;
    if (SERVER_ADMIN_ID && String(tgUser.id) === String(SERVER_ADMIN_ID)) return true;
    const fullName = [tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ').toLowerCase();
    const uname = (tgUser.username || '').toLowerCase();
    return (
      (fullName.includes('aleksandr') && fullName.includes('druzhinin')) ||
      (fullName.includes('александр') && fullName.includes('дружинин')) ||
      uname.includes('druzhinin') ||
      uname.includes('onadash') ||
      uname.includes('sniperugc')
    );
  }

  // ================= State & Pricing =================
  const STORE_KEY = 'sniper_black_v10';
  const SCRIPT_PRICE = 100;
  const TG_POST_PRICE = 75;
  const START_BONUS = 100;
  const PROMOS = { 'SNIPER': 100, 'UGC2026': 150 };

  const PACKAGES = [
    { id: 'pack_100',  credits: 100,  stars: 100, rub: '190 ₽',   usdt: '2.0 USDT',  tag: '1 сценарий' },
    { id: 'pack_300',  credits: 300,  stars: 300, rub: '490 ₽',   usdt: '5.2 USDT',  tag: 'ХИТ ВЫБОРА', hot: true },
    { id: 'pack_1000', credits: 1000, stars: 850, rub: '1 390 ₽', usdt: '14.5 USDT', tag: 'PRO -15%' }
  ];

  const FORMAT_OPTIONS = [
    { id: 'ugc',              title: 'UGC Обзор',       desc: 'Живой нативный отзыв от лица покупателя' },
    { id: 'problem_solution', title: 'Боль → Решение',  desc: 'Острая проблема ЦА и решение товаром' },
    { id: 'unboxing',         title: 'ASMR Распаковка', desc: 'Эффект первого впечатления и детали' },
    { id: 'aesthetic',        title: 'Aesthetic Визуал', desc: 'Стильный минимализм под трендовый звук' },
    { id: 'storytelling',     title: 'Сторителлинг',    desc: 'Личная история с интригой в начале' }
  ];

  const TONE_OPTIONS = [
    { id: 'native',     title: 'Нативный',     desc: 'Искренне, как голосовое лучшему другу' },
    { id: 'aggressive', title: 'Дерзкий',      desc: 'Хлёсткие триггеры и максимальное удержание' },
    { id: 'humor',      title: 'С юмором',     desc: 'Жизненная ирония и лёгкая подача' },
    { id: 'luxury',     title: 'Премиум',      desc: 'Спокойный дорогой стиль и эстетика' }
  ];

  function createDefaultState() {
    return {
      devMode: isOwnerAccount(),
      balance: START_BONUS,
      gens: 0,
      spent: 0,
      promoUsed: false,
      format: 'ugc',
      tone: 'native',
      history: []
    };
  }

  function loadAppState() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        const st = Object.assign(createDefaultState(), parsed);
        if (isOwnerAccount() && parsed.devMode === undefined) st.devMode = true;
        return st;
      }
    } catch (e) {}
    return createDefaultState();
  }

  function saveAppState() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(S));
    } catch (e) {
      S.history = S.history.slice(0, 6);
      try { localStorage.setItem(STORE_KEY, JSON.stringify(S)); } catch (e2) {}
    }
  }

  let S = loadAppState();
  let lastResultData = null;
  let livePreviewCache = null;
  const copyVault = [];
  const prompterVault = [];

  // ================= Utilities =================
  function esc(str) {
    if (str === null || str === undefined) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function saveCopyText(txt) { copyVault.push(txt == null ? '' : String(txt)); return copyVault.length - 1; }
  function copyFromVault(idx) { copyText(copyVault[idx]); }
  function savePrompterText(txt) { prompterVault.push(txt == null ? '' : String(txt)); return prompterVault.length - 1; }

  let toastTimeout = null;
  function toast(msg) {
    const el = document.getElementById('toastBanner');
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => { el.style.display = 'none'; }, 2000);
  }
  function haptic(type) {
    try { if (tg && tg.HapticFeedback) tg.HapticFeedback.notificationOccurred(type || 'success'); } catch (e) {}
  }
  function copyText(txt) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).catch(() => {});
    } else {
      const ta = document.createElement('textarea');
      ta.value = txt; ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta);
    }
    toast('Скопировано в буфер');
    haptic('success');
  }

  function openSupportChat() {
    const url = '[https://t.me/](https://t.me/)' + SUPPORT_USER;
    if (tg && tg.openTelegramLink) tg.openTelegramLink(url);
    else window.open(url, '_blank');
  }

  function shareBotReferral() {
    const shareUrl = '[https://t.me/share/url?url=](https://t.me/share/url?url=)' + encodeURIComponent('[https://t.me/sniperugc_bot](https://t.me/sniperugc_bot)') +
                     '&text=' + encodeURIComponent('🔥 AI-студия виральных сценариев для WB/Ozon со встроенным телесуфлёром! Забирай 100 ★ на старте:');
    if (tg && tg.openTelegramLink) tg.openTelegramLink(shareUrl);
    else window.open(shareUrl, '_blank');
  }

  // ================= Owner Developer Mode =================
  let logoTaps = 0;
  let logoTapTimer = null;
  function onSecretLogoTap() {
    logoTaps++;
    clearTimeout(logoTapTimer);
    if (logoTaps >= 5) {
      logoTaps = 0;
      setDevModeState(!S.devMode);
    } else {
      logoTapTimer = setTimeout(() => { logoTaps = 0; }, 1400);
    }
  }

  function setDevModeState(active) {
    S.devMode = !!active;
    saveAppState();
    syncHeaderAndPrices();
    renderCabinetView();
    toast(S.devMode ? '⚡ DEVELOPER MODE АКТИВИРОВАН' : 'Developer Mode отключён');
    haptic('success');
  }

  function devGrantStars(amt) {
    S.balance += amt;
    saveAppState();
    syncHeaderAndPrices();
    renderCabinetView();
    toast(`+${amt} ★ начислено!`);
  }

  function syncHeaderAndPrices() {
    document.getElementById('devBadgeTop').classList.toggle('active', !!S.devMode);
    document.getElementById('topBalanceVal').textContent = S.devMode ? '∞ DEV' : S.balance;
    document.getElementById('mainBtnPriceTag').textContent = S.devMode ? '(DEV ∞)' : `(${SCRIPT_PRICE} ★)`;
    document.getElementById('tgBtnPriceTag').textContent = S.devMode ? '(DEV ∞)' : `(${TG_POST_PRICE} ★)`;
    document.getElementById('studioCostHint').textContent = S.devMode
      ? '⚡ DEVELOPER MODE: безлимитная генерация сценариев и постов (0 ★)'
      : `Стоимость сценария: ${SCRIPT_PRICE} ★ · Все AI-рерайты внутри сценария бесплатны`;

    const fmtObj = FORMAT_OPTIONS.find(x => x.id === S.format) || FORMAT_OPTIONS[0];
    const toneObj = TONE_OPTIONS.find(x => x.id === S.tone) || TONE_OPTIONS[0];
    document.getElementById('activeParamsSummary').textContent = `${fmtObj.title} · ${toneObj.title}`;
  }

  // ================= Navigation =================
  function switchView(viewName) {
    ['studio', 'prompter', 'tgads', 'cabinet'].forEach(v => {
      document.getElementById('view-' + v).classList.toggle('active', v === viewName);
      document.getElementById('tabBtn-' + v).classList.toggle('active', v === viewName);
    });
    if (viewName === 'prompter') renderPrompterTab();
    if (viewName === 'tgads') renderWatermarkCanvas();
    if (viewName === 'cabinet') renderCabinetView();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function switchResultTab(paneName, btn) {
    document.querySelectorAll('.seg-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.seg-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('pane-' + paneName).classList.add('active');
  }

  // ================= Modern Settings Bottom Sheet =================
  function openSheet(html) {
    document.getElementById('sheetContent').innerHTML = `<div class="sheet-handle"></div>` + html;
    document.getElementById('sheetBackdrop').classList.add('open');
  }
  function closeSheet() {
    document.getElementById('sheetBackdrop').classList.remove('open');
  }

  function openSettingsSheet() {
    const fmtCards = FORMAT_OPTIONS.map(f => `
      <button type="button" class="opt-card ${S.format === f.id ? 'active' : ''}" onclick="selectFormatOption('${f.id}')">
        <div class="o-title"><span>${esc(f.title)}</span><span>${S.format === f.id ? '✓' : ''}</span></div>
        <div class="o-desc">${esc(f.desc)}</div>
      </button>
    `).join('');

    const toneCards = TONE_OPTIONS.map(t => `
      <button type="button" class="opt-card ${S.tone === t.id ? 'active' : ''}" onclick="selectToneOption('${t.id}')">
        <div class="o-title"><span>${esc(t.title)}</span><span>${S.tone === t.id ? '✓' : ''}</span></div>
        <div class="o-desc">${esc(t.desc)}</div>
      </button>
    `).join('');

    openSheet(`
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
        <div style="font-weight:800; font-size:17px;">⚙️ Параметры сценария</div>
        <button class="btn-chip" onclick="closeSheet()">Готово</button>
      </div>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">01 / Формат видеоролика</div>
      <div class="opt-grid">${fmtCards}</div>
      <div class="mono" style="font-size:10.5px; color:var(--muted); text-transform:uppercase; margin-bottom:8px;">02 / Тональность и вайб</div>
      <div class="opt-grid">${toneCards}</div>
      <button class="btn-primary" onclick="closeSheet()">ПРИМЕНИТЬ НАСТРОЙКИ</button>
    `);
  }

  function selectFormatOption(id) {
    S.format = id;
    saveAppState();
    syncHeaderAndPrices();
    openSettingsSheet();
    haptic('success');
  }

  function selectToneOption(id) {
    S.tone = id;
    saveAppState();
    syncHeaderAndPrices();
    openSettingsSheet();
    haptic('success');
  }

  // ================= Live Product Overview (Instant Link Preview) =================
  let previewTimers = {};
  function onProductInput(el, mode) {
    const val = (el.value || '').trim();
    const clearBtn = document.getElementById(mode === 'main' ? 'clearMainBtn' : 'clearTgBtn');
    if (clearBtn) clearBtn.style.display = val.length > 0 ? 'inline-flex' : 'none';

    const mirrorId = mode === 'main' ? 'tgQueryInp' : 'mainQueryInp';
    const mirrorEl = document.getElementById(mirrorId);
    if (mirrorEl && !mirrorEl.value.trim()) mirrorEl.value = val;

    clearTimeout(previewTimers[mode]);
    const card = document.getElementById(mode === 'main' ? 'overviewMain' : 'overviewTg');
    if (val.length < 5) {
      if (card) card.style.display = 'none';
      return;
    }
    previewTimers[mode] = setTimeout(() => fetchProductOverview(val, mode), 380);
  }

  async function pasteFromClipboard(inpId, mode) {
    const el = document.getElementById(inpId);
    try {
      if (navigator.clipboard && navigator.clipboard.readText) {
        const clip = await navigator.clipboard.readText();
        if (clip) {
          el.value = clip.trim();
          onProductInput(el, mode);
          haptic('success');
          return;
        }
      }
    } catch (e) {}
    el.focus();
  }

  function clearProductInput(inpId, mode) {
    const el = document.getElementById(inpId);
    el.value = '';
    onProductInput(el, mode);
    const card = document.getElementById(mode === 'main' ? 'overviewMain' : 'overviewTg');
    if (card) card.style.display = 'none';
  }

  async function fetchProductOverview(query, mode) {
    const card = document.getElementById(mode === 'main' ? 'overviewMain' : 'overviewTg');
    if (!card) return;
    try {
      const resp = await fetch('/api/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query })
      });
      if (!resp.ok) return;
      const d = await resp.json();
      livePreviewCache = d;

      const thumb = d.image ? `<img class="ov-img" src="${esc(d.image)}" alt="" onerror="this.style.display='none'">` : '';
      const price = d.price ? `<span class="ov-price">${esc(d.price)}</span>` : '';
      const meta = [d.category, d.brand].filter(Boolean).join(' · ') || 'Карточка товара распознана';

      card.innerHTML = `
        <div class="ov-flex">
          ${thumb}
          <div class="ov-info">
            <div class="ov-badges">
              <span class="ov-source">✓ ${esc(d.source || 'ТОВАР')}</span>
              ${price}
            </div>
            <div class="ov-title">${esc(d.title)}</div>
            <div class="ov-meta">${esc(meta)}</div>
          </div>
        </div>
      `;
      card.style.display = 'block';
      if (d.image) loadWatermarkImageFromUrl(d.image);
    } catch (e) {}
  }

  // ================= AI Neural Streaming Animation =================
  let neuralTimer = null;
  function startNeuralConsole(rawQuery) {
    const box = document.getElementById('neuralBox');
    const bar = document.getElementById('nbBarFill');
    const pct = document.getElementById('nbPctTxt');
    const log = document.getElementById('nbLogStream');
    box.style.display = 'block';
    log.innerHTML = '';

    const title = (livePreviewCache && livePreviewCache.title) ? livePreviewCache.title : rawQuery.slice(0, 44);
    const src = (livePreviewCache && livePreviewCache.source) ? livePreviewCache.source : 'Источник';

    const steps = [
      { p: 18, t: `Загружена карточка ${src}: «${title}»` },
      { p: 38, t: `Очистка от посторонних рекомендаций и выделение реальных характеристик...` },
      { p: 58, t: `Генерация 5 виральных хуков на русском языке (удержание 0-2 сек)...` },
      { p: 78, t: `Сборка покадрового сценария, текста для суфлёра и звука...` },
      { p: 94, t: `Написание 3 нативных рекламных постов для Telegram...` }
    ];

    let i = 0;
    function step() {
      if (i >= steps.length) return;
      const s = steps[i];
      bar.style.width = s.p + '%';
      pct.textContent = s.p + '%';
      const row = document.createElement('div');
      row.className = 'nb-line';
      row.innerHTML = `<span class="arrow">▸</span><span>${esc(s.t)}</span>`;
      log.appendChild(row);
      i++;
    }
    step();
    clearInterval(neuralTimer);
    neuralTimer = setInterval(step, 2100);
  }

  function stopNeuralConsole() {
    clearInterval(neuralTimer);
    document.getElementById('neuralBox').style.display = 'none';
  }

  // ================= Main Pipeline Execution =================
  async function runFullAnalysis() {
    const errEl = document.getElementById('studioErrorBox');
    errEl.style.display = 'none';

    const q = document.getElementById('mainQueryInp').value.trim();
    if (!q) {
      toast('Вставь ссылку на товар или артикул');
      haptic('error');
      return;
    }
    if (!S.devMode && S.balance < SCRIPT_PRICE) {
      openCheckoutSheet(0);
      return;
    }

    const btn = document.getElementById('generateMainBtn');
    const resArea = document.getElementById('resultsArea');
    btn.disabled = true;
    resArea.style.display = 'none';
    startNeuralConsole(q);

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, format_type: S.format, tone: S.tone, lang: 'ru' })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Ошибка генерации');

      if (!S.devMode) {
        S.balance -= SCRIPT_PRICE;
        S.spent += SCRIPT_PRICE;
      }
      S.gens += 1;
      S.history.unshift({
        id: Date.now(),
        ts: Date.now(),
        title: data.product_title || q.slice(0, 40),
        query: q.slice(0, 60),
        data: data
      });
      S.history = S.history.slice(0, 15);
      saveAppState();
      syncHeaderAndPrices();

      lastResultData = data;
      renderStudioResults(data);
      resArea.style.display = 'block';
      if (data.meta_image) loadWatermarkImageFromUrl(data.meta_image);
      haptic('success');
      setTimeout(() => resArea.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } catch (e) {
      errEl.textContent = '⚠️ ' + (e.message || 'Не удалось сгенерировать сценарий');
      errEl.style.display = 'block';
      haptic('error');
    } finally {
      stopNeuralConsole();
      btn.disabled = false;
    }
  }

  // ================= Render Studio Results + 1-Tap AI Rewrite =================
  function renderStudioResults(d) {
    copyVault.length = 0;
    prompterVault.length = 0;

    const thumb = d.meta_image ? `<img class="ov-img" src="${esc(d.meta_image)}" alt="" onerror="this.style.display='none'">` : '';
    const price = d.meta_price ? `<span class="mini-tag neon">${esc(d.meta_price)}</span>` : '';
    document.getElementById('resultProductCard').innerHTML = `
      <div class="ov-flex">
        ${thumb}
        <div class="ov-info">
          <div style="font-weight:800; font-size:15px; margin-bottom:4px;">${esc(d.product_title)}</div>
          <div style="display:flex; gap:6px; flex-wrap:wrap;">
            <span class="mini-tag">${esc(d.meta_source || 'Товар')}</span>
            <span class="mini-tag">${esc(d.category || '')}</span>
            ${price}
          </div>
        </div>
      </div>
    `;

    // 1. SCRIPTS PANE
    document.getElementById('pane-scripts').innerHTML = (d.scripts || []).map((sc, i) => {
      const fullCopy = `СЦЕНАРИЙ: ${sc.title}\nХУК: ${sc.hook}\n` +
        (sc.scenes || []).map(s => `[${s.timecode}] Кадр: ${s.visual} | Озвучка: ${s.voiceover}`).join('\n') +
        `\nCTA: ${sc.cta}`;

      const voiceOnly = [sc.hook, ...(sc.scenes || []).map(s => s.voiceover), sc.cta].filter(Boolean).join('\n\n');
      const tpIdx = savePrompterText(voiceOnly);
      const cpIdx = saveCopyText(fullCopy);

      const scenesHtml = (sc.scenes || []).map(s => `
        <div class="scene-block">
          <div class="scene-time">⏱ ${esc(s.timecode)}</div>
          <div><strong style="color:var(--muted);">Кадр:</strong> ${esc(s.visual)}</div>
          <div style="margin:4px 0; color:#fff;"><strong style="color:var(--neon);">Озвучка:</strong> «${esc(s.voiceover)}»</div>
          <div><strong style="color:var(--muted);">Плашка:</strong> ${esc(s.on_screen_text)}</div>
          <div class="prompt-strip">
            <span>🎥 AI: ${esc(s.ai_video_prompt)}</span>
            <button class="btn-chip" onclick="copyFromVault(${saveCopyText(s.ai_video_prompt)})">COPY</button>
          </div>
        </div>
      `).join('');

      const soundBadge = sc.sound_vibe ? `<span class="mini-tag">🎵 ${esc(sc.sound_vibe)}</span>` : '';

      return `
        <div class="bento-card">
          <div class="script-head">
            <div style="min-width:0;">
              <div style="font-weight:800; font-size:15px;">0${i + 1} // ${esc(sc.title)}</div>
              <div class="script-badge-row">
                <span class="mini-tag neon">${esc(sc.virality_score || 96)}% виральность</span>
                <span class="mini-tag">${esc(sc.angle)}</span>
                ${soundBadge}
              </div>
            </div>
            <div style="display:flex; gap:6px; flex-shrink:0;">
              <button class="btn-chip accent" onclick="openPrompterWithVault(${tpIdx})">📟 В СУФЛЁР</button>
              <button class="btn-chip" onclick="copyFromVault(${cpIdx})">📋</button>
            </div>
          </div>

          <div class="hook-callout"><span class="mono" style="font-size:10px; color:var(--neon); display:block;">ХУК (0-2 СЕК):</span>«${esc(sc.hook)}»</div>
          ${scenesHtml}
          <div style="font-size:13px; margin-top:8px;"><strong style="color:var(--neon);">Призыв (CTA):</strong> ${esc(sc.cta)}</div>

          <!-- 1-TAP AI REWRITE TOOLBAR -->
          <div class="rw-box">
            <div class="rw-title">✨ Мгновенный AI-рерайт сценария (Бесплатно):</div>
            <div class="rw-grid">
              <button class="rw-pill" onclick="triggerScriptRewrite(${i}, 'shorter')">✂️ Короче</button>
              <button class="rw-pill" onclick="triggerScriptRewrite(${i}, 'longer')">📝 Подробнее</button>
              <button class="rw-pill" onclick="triggerScriptRewrite(${i}, 'bolder')">🔥 Дерзко</button>
              <button class="rw-pill" onclick="triggerScriptRewrite(${i}, 'native')">💬 Нативнее</button>
              <button class="rw-pill" onclick="triggerScriptRewrite(${i}, 'funny')">😂 С юмором</button>
              <button class="rw-pill" onclick="toggleCustomRwInput(${i})">✍️ Своя правка</button>
            </div>
            <div id="customRwRow-${i}" style="display:none; gap:6px; margin-top:8px;">
              <input id="customRwInp-${i}" class="field-inp" placeholder="Напр.: сделай от лица девушки / добавь скидку 50%...">
              <button class="btn-primary" style="width:auto; padding:0 14px;" onclick="triggerScriptRewrite(${i}, 'custom')">ОК</button>
            </div>
            <div id="rwLoadingTxt-${i}" class="mono" style="display:none; font-size:11px; color:var(--neon); margin-top:8px;">⚡ ИИ переписывает сценарий...</div>
          </div>
        </div>
      `;
    }).join('');

    // 2. TG POSTS PANE
    document.getElementById('pane-tgposts').innerHTML = renderTgPostCardsHtml(d.tg_posts || []);

    // 3. HOOKS PANE
    document.getElementById('pane-hooks').innerHTML = `
      <div class="bento-card">
        ${(d.viral_hooks || []).map((hk, idx) => `
          <div style="display:flex; justify-content:space-between; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid var(--border);">
            <div style="font-size:13.5px;"><strong style="color:var(--neon);">#0${idx + 1}</strong> «${esc(hk)}»</div>
            <button class="btn-chip" onclick="copyFromVault(${saveCopyText(hk)})">COPY</button>
          </div>
        `).join('')}
      </div>
    `;

    // 4. AI VIDEO PROMPTS
    document.getElementById('pane-broll').innerHTML = `
      <div class="bento-card">
        ${(d.video_generator_prompts || []).map(p => `
          <div class="prompt-strip" style="margin-bottom:8px;">
            <span>${esc(p)}</span>
            <button class="btn-chip" onclick="copyFromVault(${saveCopyText(p)})">COPY</button>
          </div>
        `).join('')}
      </div>
    `;

    // 5. AUDIENCE & SEO
    const tags = (d.hashtags || []).join(' ');
    const fullSeo = `${d.seo_caption || ''}\n\n${tags}`;
    document.getElementById('pane-seo').innerHTML = `
      <div class="bento-card">
        <div style="font-weight:800; margin-bottom:6px;">Целевая аудитория</div>
        <div style="font-size:13.5px; color:#d4d4e0; margin-bottom:12px;">${esc(d.target_audience)}</div>
        <div style="font-weight:800; margin-bottom:6px;">Реальные преимущества товара</div>
        <ul style="padding-left:18px; font-size:13px; color:#d4d4e0; margin-bottom:12px;">
          ${(d.killer_features || []).map(f => `<li>${esc(f)}</li>`).join('')}
        </ul>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
          <span style="font-weight:800;">SEO-описание и теги</span>
          <button class="btn-chip" onclick="copyFromVault(${saveCopyText(fullSeo)})">СКОПИРОВАТЬ SEO</button>
        </div>
        <div style="font-size:13px; margin-bottom:6px;">${esc(d.seo_caption)}</div>
        <div class="mono" style="font-size:11.5px; color:var(--neon);">${esc(tags)}</div>
      </div>
    `;
  }

  function toggleCustomRwInput(idx) {
    const row = document.getElementById('customRwRow-' + idx);
    if (!row) return;
    row.style.display = row.style.display === 'flex' ? 'none' : 'flex';
  }

  async function triggerScriptRewrite(idx, action) {
    if (!lastResultData || !lastResultData.scripts || !lastResultData.scripts[idx]) return;
    let customCmd = '';
    if (action === 'custom') {
      const inp = document.getElementById('customRwInp-' + idx);
      customCmd = (inp && inp.value) ? inp.value.trim() : '';
      if (!customCmd) { toast('Введи свою команду для рерайта'); return; }
    }

    const loader = document.getElementById('rwLoadingTxt-' + idx);
    if (loader) loader.style.display = 'block';
    haptic('warning');

    try {
      const resp = await fetch('/api/rewrite-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          product_title: lastResultData.product_title || '',
          script: lastResultData.scripts[idx],
          action: action,
          custom_instruction: customCmd,
          lang: 'ru'
        })
      });
      const data = await resp.json();
      if (!resp.ok || !data.script) throw new Error(data.detail || 'Ошибка рерайта');

      lastResultData.scripts[idx] = data.script;
      if (S.history.length > 0) {
        S.history[0].data = lastResultData;
        saveAppState();
      }
      renderStudioResults(lastResultData);
      toast('✨ Сценарий успешно переписан!');
      haptic('success');
    } catch (e) {
      toast(e.message || 'Ошибка рерайта');
      if (loader) loader.style.display = 'none';
    }
  }

  // ================= Dedicated Teleprompter Tab & Player =================
  let tpSpeed = 1.2;
  let tpFont = 28;
  let tpMirror = false;
  let tpPlaying = false;
  let tpRafId = null;
  let tpLastTime = 0;

  function renderPrompterTab() {
    const listEl = document.getElementById('prompterHistoryList');
    const items = [];
    S.history.forEach((hItem, hIdx) => {
      const scripts = (hItem.data && hItem.data.scripts) || [];
      scripts.forEach((sc, sIdx) => {
        const text = [sc.hook, ...(sc.scenes || []).map(x => x.voiceover), sc.cta].filter(Boolean).join('\n\n');
        items.push({
          title: `${hItem.title} — ${sc.title}`,
          angle: sc.angle || 'Сценарий',
          text: text
        });
      });
    });

    if (!items.length) {
      listEl.innerHTML = `<div style="padding:12px; border:1px dashed var(--border-light); border-radius:12px; color:var(--muted); font-size:12.5px; text-align:center;">Пока нет сохранённых сценариев. Создай сценарий на первой вкладке или вставь свой текст ниже.</div>`;
      return;
    }

    listEl.innerHTML = items.slice(0, 8).map((it, idx) => {
      const vIdx = savePrompterText(it.text);
      return `
        <div class="tp-script-pick" onclick="loadScriptIntoPrompterTab(${vIdx}, this)">
          <div style="min-width:0;">
            <div style="font-weight:700; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">🎬 ${esc(it.title)}</div>
            <div class="mono" style="font-size:10.5px; color:var(--muted);">${esc(it.angle)}</div>
          </div>
          <span class="btn-chip accent">Выбрать</span>
        </div>
      `;
    }).join('');

    const editor = document.getElementById('prompterEditorInp');
    if (!editor.value.trim() && items[0]) {
      editor.value = items[0].text;
    }
  }

  function loadScriptIntoPrompterTab(vIdx, cardEl) {
    document.querySelectorAll('.tp-script-pick').forEach(el => el.classList.remove('selected'));
    if (cardEl) cardEl.classList.add('selected');
    document.getElementById('prompterEditorInp').value = prompterVault[vIdx] || '';
    toast('Текст загружен в суфлёр');
    haptic('success');
  }

  function openPrompterWithVault(vIdx) {
    const txt = prompterVault[vIdx] || '';
    document.getElementById('prompterEditorInp').value = txt;
    openFullscreenPrompterModal(txt);
  }

  function launchFullscreenPrompterFromTab() {
    const txt = document.getElementById('prompterEditorInp').value.trim();
    if (!txt) {
      toast('Сначала выбери сценарий или введи текст');
      return;
    }
    openFullscreenPrompterModal(txt);
  }

  function syncTpSpeed(val) {
    tpSpeed = parseFloat(val) || 1.2;
    document.getElementById('tpTabSpeedVal').textContent = tpSpeed.toFixed(1) + 'x';
    document.getElementById('tpModalSpeedLbl').textContent = tpSpeed.toFixed(1) + 'x';
    document.getElementById('tpModalSpeedRange').value = tpSpeed;
  }

  function syncTpFont(val) {
    tpFont = parseInt(val, 10) || 28;
    document.getElementById('tpTabFontVal').textContent = tpFont + 'px';
    document.getElementById('tpLiveTextNode').style.fontSize = tpFont + 'px';
  }

  function nudgePrompterFont(delta) {
    syncTpFont(Math.min(46, Math.max(18, tpFont + delta)));
  }

  function openFullscreenPrompterModal(text) {
    stopPrompterScroll();
    const node = document.getElementById('tpLiveTextNode');
    node.textContent = text;
    node.style.fontSize = tpFont + 'px';
    document.getElementById('tpScrollStage').scrollTop = 0;
    document.getElementById('tpFullscreenPlayer').classList.add('open');
  }

  function closeFullscreenPrompter() {
    stopPrompterScroll();
    document.getElementById('tpFullscreenPlayer').classList.remove('open');
  }

  function resetPrompterScroll() {
    stopPrompterScroll();
    document.getElementById('tpScrollStage').scrollTop = 0;
  }

  function togglePrompterMirror() {
    tpMirror = !tpMirror;
    document.getElementById('tpScrollStage').classList.toggle('mirror', tpMirror);
    document.getElementById('tpMirrorToggleBtn').classList.toggle('accent', tpMirror);
  }

  function startPrompterWithCountdown() {
    if (tpPlaying) { stopPrompterScroll(); return; }
    const cd = document.getElementById('tpCountdownBox');
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
        beginPrompterScroll();
      }
    }, 800);
  }

  function togglePrompterPlayPause() {
    if (document.activeElement === document.getElementById('tpLiveTextNode')) return;
    if (tpPlaying) stopPrompterScroll();
    else beginPrompterScroll();
  }

  function beginPrompterScroll() {
    tpPlaying = true;
    tpLastTime = performance.now();
    document.getElementById('tpPlayActionBtn').textContent = '⏸ ПАУЗА (ТАП ПО ЭКРАНУ)';
    const stage = document.getElementById('tpScrollStage');
    function frame(now) {
      if (!tpPlaying) return;
      const dt = (now - tpLastTime) / 1000;
      tpLastTime = now;
      stage.scrollTop += (40 * tpSpeed) * dt;
      if (stage.scrollTop + stage.clientHeight >= stage.scrollHeight - 4) {
        stopPrompterScroll();
        return;
      }
      tpRafId = requestAnimationFrame(frame);
    }
    tpRafId = requestAnimationFrame(frame);
  }

  function stopPrompterScroll() {
    tpPlaying = false;
    if (tpRafId) cancelAnimationFrame(tpRafId);
    const btn = document.getElementById('tpPlayActionBtn');
    if (btn) btn.textContent = '▶ СТАРТ / ПРОДОЛЖИТЬ';
  }

  // ================= TG Ad Posts + ERID + Watermark =================
  function buildEridFooter() {
    const org = (document.getElementById('eridOrgInp')?.value || '').trim();
    const tok = (document.getElementById('eridTokenInp')?.value || '').trim().replace(/^erid:\s*/i, '');
    const link = (document.getElementById('eridUrlInp')?.value || '').trim();
    const parts = [];
    if (link) parts.push(`👉 Заказать / Артикул: ${link}`);
    if (org || tok) {
      parts.push(`Реклама. ${org || 'Рекламодатель'}${tok ? `, erid: ${tok}` : ''}`);
    }
    return parts.join('\n');
  }

  function refreshEridFooters() {
    const footer = buildEridFooter();
    const box = document.getElementById('eridLivePreview');
    const tok = (document.getElementById('eridTokenInp')?.value || '').trim().replace(/^erid:\s*/i, '');
    if (tok) document.getElementById('wmEridStampInp').value = `Реклама • erid: ${tok}`;
    if (footer) {
      box.style.display = 'block';
      box.textContent = footer;
    } else {
      box.style.display = 'none';
    }
    renderWatermarkCanvas();
  }

  function renderTgPostCardsHtml(posts) {
    const footer = buildEridFooter();
    return (posts || []).map((p, i) => {
      const fullTxt = `${p.headline}\n\n${p.body}\n\n${p.cta}${footer ? '\n\n' + footer : ''}`;
      const cpIdx = saveCopyText(fullTxt);
      const tpIdx = savePrompterText(`${p.headline}\n\n${p.body}`);
      return `
        <div class="bento-card">
          <div class="script-head">
            <span class="mini-tag neon">ПОСТ #0${i + 1} · ${esc(p.style)}</span>
            <div style="display:flex; gap:6px;">
              <button class="btn-chip" onclick="openPrompterWithVault(${tpIdx})">📟</button>
              <button class="btn-chip accent" onclick="copyFromVault(${cpIdx})">📋 СКОПИРОВАТЬ ПОСТ</button>
            </div>
          </div>
          <div style="font-weight:800; font-size:15px; margin-bottom:8px;">${esc(p.headline)}</div>
          <div style="white-space:pre-line; font-size:13.5px; color:#e4e4ef; margin-bottom:10px;">${esc(p.body)}</div>
          <div style="font-weight:700; font-size:13px; color:var(--neon);">${esc(p.cta)}</div>
          ${footer ? `<div class="erid-tag-box">${esc(footer)}</div>` : ''}
        </div>
      `;
    }).join('');
  }

  async function runStandaloneTgPosts() {
    const q = document.getElementById('tgQueryInp').value.trim();
    if (!q) { toast('Вставь ссылку на товар'); return; }
    if (!S.devMode && S.balance < TG_POST_PRICE) { openCheckoutSheet(0); return; }

    const btn = document.getElementById('tgGenerateBtn');
    const out = document.getElementById('tgPostsListArea');
    btn.disabled = true;
    out.innerHTML = `<div class="bento-card mono" style="text-align:center; color:var(--neon);">⚡ Генерирую 3 рекламных поста по характеристикам товара...</div>`;

    try {
      const resp = await fetch('/api/tg-post', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, style: S.tone, lang: 'ru' })
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Ошибка генерации постов');

      if (!S.devMode) {
        S.balance -= TG_POST_PRICE;
        S.spent += TG_POST_PRICE;
      }
      S.gens += 1;
      saveAppState();
      syncHeaderAndPrices();

      if (data.meta_image) loadWatermarkImageFromUrl(data.meta_image);
      out.innerHTML = renderTgPostCardsHtml(data.tg_posts || []);
      haptic('success');
    } catch (e) {
      out.innerHTML = `<div class="bento-card" style="border-color:var(--danger); color:#fda4af;">⚠️ ${esc(e.message)}</div>`;
    } finally {
      btn.disabled = false;
    }
  }

  // Watermark Studio
  let wmPosition = 'br';
  let wmLoadedImg = null;

  function setWmPosition(pos, btn) {
    wmPosition = pos;
    btn.parentElement.querySelectorAll('.btn-chip').forEach(b => b.classList.remove('accent'));
    btn.classList.add('accent');
    renderWatermarkCanvas();
  }

  function loadWatermarkImageFromUrl(url) {
    if (!url) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => { wmLoadedImg = img; renderWatermarkCanvas(); };
    img.src = '/api/proxy-image?url=' + encodeURIComponent(url);
  }

  function onCustomWatermarkFile(ev) {
    const f = ev.target.files && ev.target.files[0];
    if (!f) return;
    const r = new FileReader();
    r.onload = e => {
      const img = new Image();
      img.onload = () => { wmLoadedImg = img; renderWatermarkCanvas(); };
      img.src = e.target.result;
    };
    r.readAsDataURL(f);
  }

  function renderWatermarkCanvas() {
    const canvas = document.getElementById('wmCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = '#07070b';
    ctx.fillRect(0, 0, W, H);

    if (wmLoadedImg) {
      const ratio = Math.max(W / wmLoadedImg.width, H / wmLoadedImg.height);
      const nw = wmLoadedImg.width * ratio, nh = wmLoadedImg.height * ratio;
      ctx.drawImage(wmLoadedImg, (W - nw) / 2, (H - nh) / 2, nw, nh);
    } else {
      ctx.strokeStyle = 'rgba(204,255,0,0.2)';
      ctx.strokeRect(24, 24, W - 48, H - 48);
      ctx.fillStyle = '#6b6b7b';
      ctx.font = '600 18px Plus Jakarta Sans, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Фото товара подгрузится из ссылки', W / 2, H / 2 - 8);
      ctx.fillText('или нажми «📷 Загрузить фото»', W / 2, H / 2 + 20);
    }

    const handle = (document.getElementById('wmHandleInp')?.value || '').trim();
    const erid = (document.getElementById('wmEridStampInp')?.value || '').trim();

    if (handle) {
      ctx.save();
      if (wmPosition === 'center') {
        ctx.translate(W / 2, H / 2);
        ctx.rotate(-Math.PI / 6);
        ctx.font = '800 40px Syne, sans-serif';
        ctx.fillStyle = 'rgba(255,255,255,0.28)';
        ctx.textAlign = 'center';
        ctx.fillText(handle, 0, 0);
      } else {
        ctx.font = '700 20px Plus Jakarta Sans, sans-serif';
        const tw = ctx.measureText(handle).width;
        const bw = tw + 32, bh = 38;
        let bx = W - bw - 20, by = H - bh - 20;
        if (wmPosition === 'bl') bx = 20;
        if (wmPosition === 'tl') { bx = 20; by = 20; }
        ctx.fillStyle = 'rgba(5,5,8,0.78)';
        ctx.beginPath();
        ctx.roundRect(bx, by, bw, bh, 10);
        ctx.fill();
        ctx.fillStyle = '#ccff00';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(handle, bx + 16, by + bh / 2);
      }
      ctx.restore();
    }

    if (erid) {
      ctx.save();
      ctx.font = '600 14px JetBrains Mono, monospace';
      const tw = ctx.measureText(erid).width;
      const bw = tw + 20, bh = 28;
      const bx = 20, by = H - bh - (wmPosition === 'bl' ? 68 : 20);
      ctx.fillStyle = 'rgba(0,0,0,0.8)';
      ctx.beginPath();
      ctx.roundRect(bx, by, bw, bh, 7);
      ctx.fill();
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(erid, bx + 10, by + bh / 2);
      ctx.restore();
    }
  }

  function downloadWatermarkPng() {
    const canvas = document.getElementById('wmCanvas');
    const a = document.createElement('a');
    a.href = canvas.toDataURL('image/png');
    a.download = 'sniper_ugc_product.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    toast('Фото скачано');
  }

  // ================= Cabinet & Multi-Method Demo Checkout (Stars / SBP / Crypto) =================
  let activePayMethod = 'stars'; // stars, sbp, crypto
  let selectedPackIdx = 1;

  function renderCabinetView() {
    const name = tgUser ? ([tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ') || tgUser.username || 'Креатор') : 'Креатор';
    const sub = tgUser && tgUser.username ? '@' + tgUser.username : 'SNIPER STUDIO PRO';
    const badge = S.devMode
      ? `<span class="mini-tag neon">⚡ OWNER // DEV MODE</span>`
      : `<span class="mini-tag">PRO КРЕАТОР</span>`;

    document.getElementById('cabProfileHeader').innerHTML = `
      <div class="brand-mark" style="width:46px; height:46px; font-size:16px; border-radius:50%;">${esc(name.charAt(0).toUpperCase())}</div>
      <div style="min-width:0;">
        <div style="font-weight:800; font-size:16px;">${esc(name)}</div>
        <div style="display:flex; gap:6px; align-items:center; margin-top:3px;">
          <span style="font-size:12px; color:var(--muted);">${esc(sub)}</span>
          ${badge}
        </div>
      </div>
    `;

    document.getElementById('ownerDevConsole').style.display = S.devMode ? 'block' : 'none';
    document.getElementById('cabBigBalance').textContent = S.devMode ? '∞ DEV' : `${S.balance} ★`;

    document.getElementById('cabPackCards').innerHTML = PACKAGES.map((p, i) => `
      <div onclick="openCheckoutSheet(${i})" style="background:#07070b; border:1.5px solid ${p.hot ? 'var(--neon)' : 'var(--border-light)'}; border-radius:14px; padding:11px 6px; text-align:center; cursor:pointer;">
        <div class="mono" style="font-size:9px; color:${p.hot ? 'var(--neon)' : 'var(--muted)'}; margin-bottom:3px;">${esc(p.tag)}</div>
        <div style="font-family:var(--display); font-size:18px; font-weight:800;">+${p.credits} ★</div>
        <div style="font-size:11px; color:var(--muted); margin-top:2px;">${esc(p.rub)}</div>
      </div>
    `).join('');

    const histBox = document.getElementById('cabHistoryContainer');
    if (!S.history.length) {
      histBox.innerHTML = `<div style="color:var(--muted); font-size:12.5px; text-align:center; padding:12px;">История пока пуста</div>`;
    } else {
      histBox.innerHTML = S.history.map((it, idx) => `
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; padding:10px 0; border-bottom:1px solid var(--border);">
          <div style="min-width:0;">
            <div style="font-weight:700; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(it.title)}</div>
            <div class="mono" style="font-size:10.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">${esc(it.query)}</div>
          </div>
          <button class="btn-chip accent" onclick="loadHistoryProject(${idx})">Открыть</button>
        </div>
      `).join('');
    }
  }

  function loadHistoryProject(idx) {
    const item = S.history[idx];
    if (!item) return;
    lastResultData = item.data;
    renderStudioResults(item.data);
    switchView('studio');
    document.getElementById('resultsArea').style.display = 'block';
  }

  function wipeHistory() {
    S.history = [];
    saveAppState();
    renderCabinetView();
    toast('История очищена');
  }

  function openCheckoutSheet(packIdx) {
    if (typeof packIdx === 'number') selectedPackIdx = packIdx;
    const p = PACKAGES[selectedPackIdx] || PACKAGES[1];

    const packRow = PACKAGES.map((pk, idx) => `
      <button type="button" onclick="openCheckoutSheet(${idx})" style="flex:1; background:${idx === selectedPackIdx ? 'var(--neon-soft)' : '#07070b'}; border:1.5px solid ${idx === selectedPackIdx ? 'var(--neon)' : 'var(--border)'}; color:var(--text); border-radius:11px; padding:8px 4px; cursor:pointer;">
        <div style="font-weight:800; font-size:13px;">+${pk.credits} ★</div>
        <div class="mono" style="font-size:10px; color:var(--muted);">${activePayMethod === 'stars' ? pk.stars + ' ★' : (activePayMethod === 'sbp' ? pk.rub : pk.usdt)}</div>
      </button>
    `).join('');

    let methodBody = '';
    if (activePayMethod === 'stars') {
      methodBody = `
        <div style="background:#07070b; border:1px solid var(--border); border-radius:14px; padding:14px; margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="color:var(--muted);">К зачислению:</span>
            <strong style="color:var(--neon);">+${p.credits} ★</strong>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span style="color:var(--muted);">Сумма списания:</span>
            <strong>${p.stars} Telegram Stars</strong>
          </div>
        </div>
        <button id="starsPayExecBtn" class="btn-primary" onclick="executeTelegramStarsPayment(${selectedPackIdx})">
          ⭐ ОПЛАТИТЬ ${p.stars} STARS
        </button>
      `;
    } else if (activePayMethod === 'sbp') {
      methodBody = `
        <div style="background:#07070b; border:1px solid var(--border); border-radius:14px; padding:14px; margin-bottom:14px; text-align:center;">
          <div class="mono" style="font-size:10.5px; color:var(--neon); margin-bottom:4px;">СБП БЫСТРЫЙ ПЛАТЁЖ · КАРТЫ РФ (МИР)</div>
          <div style="font-size:18px; font-weight:800; margin-bottom:6px;">К оплате: ${esc(p.rub)} за +${p.credits} ★</div>
          <div class="qr-mock">
            <svg viewBox="0 0 100 100" width="124" height="124">
              <rect width="30" height="30" x="5" y="5" fill="#000"/><rect width="18" height="18" x="11" y="11" fill="#fff"/><rect width="10" height="10" x="15" y="15" fill="#000"/>
              <rect width="30" height="30" x="65" y="5" fill="#000"/><rect width="18" height="18" x="71" y="11" fill="#fff"/><rect width="10" height="10" x="75" y="15" fill="#000"/>
              <rect width="30" height="30" x="5" y="65" fill="#000"/><rect width="18" height="18" x="11" y="71" fill="#fff"/><rect width="10" height="10" x="15" y="75" fill="#000"/>
              <rect width="12" height="12" x="44" y="12" fill="#000"/><rect width="12" height="24" x="44" y="36" fill="#000"/><rect width="20" height="12" x="64" y="48" fill="#000"/><rect width="24" height="16" x="48" y="74" fill="#000"/>
            </svg>
          </div>
          <div style="font-size:11.5px; color:var(--muted);">Демо-шлюз СБП: нажми кнопку ниже для мгновенного тестового зачисления</div>
        </div>
        <button class="btn-primary" onclick="completeDemoPayment(${p.credits}, 'СБП / Карта РФ')">
          ⚡ ПОДТВЕРДИТЬ ОПЛАТУ СБП (${esc(p.rub)})
        </button>
      `;
    } else {
      methodBody = `
        <div style="background:#07070b; border:1px solid var(--border); border-radius:14px; padding:14px; margin-bottom:14px;">
          <div class="mono" style="font-size:10.5px; color:var(--neon); margin-bottom:6px;">USDT (TRC-20 / TON) · CRYPTO PAY</div>
          <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
            <span style="color:var(--muted);">Сумма к переводу:</span>
            <strong style="color:var(--neon);">${esc(p.usdt)} (+${p.credits} ★)</strong>
          </div>
          <div class="mono" style="font-size:11px; background:var(--card); padding:9px; border-radius:8px; border:1px solid var(--border-light); word-break:break-all; margin-bottom:8px;">
            TQn9Y2khEsLJW1ChVWFMSMeRDow5KcbLSE
          </div>
          <div style="font-size:11.5px; color:var(--muted);">Демо-режим крипто-кассы: средства зачисляются по кнопке проверки транзакции</div>
        </div>
        <button class="btn-primary" onclick="completeDemoPayment(${p.credits}, 'USDT Crypto')">
          💎 Я ОПЛАТИЛ — ПРОВЕРИТЬ ТРАНЗАКЦИЮ
        </button>
      `;
    }

    openSheet(`
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
        <div style="font-weight:800; font-size:17px;">Пополнение баланса</div>
        <button class="btn-chip" onclick="closeSheet()">✕</button>
      </div>

      <div class="pay-methods">
        <button type="button" class="pm-btn ${activePayMethod === 'stars' ? 'active' : ''}" onclick="switchPayMethod('stars')">★ TG Stars</button>
        <button type="button" class="pm-btn ${activePayMethod === 'sbp' ? 'active' : ''}" onclick="switchPayMethod('sbp')">💳 СБП / РФ</button>
        <button type="button" class="pm-btn ${activePayMethod === 'crypto' ? 'active' : ''}" onclick="switchPayMethod('crypto')">💎 USDT / TON</button>
      </div>

      <div style="display:flex; gap:6px; margin-bottom:14px;">${packRow}</div>
      ${methodBody}

      <div style="text-align:center; margin-top:12px;">
        <button type="button" style="background:none; border:none; color:var(--muted); font-size:12px; cursor:pointer; text-decoration:underline;" onclick="openSupportChat()">
          Вопросы по оплате? Саппорт @onadash
        </button>
      </div>
    `);
  }

  function switchPayMethod(m) {
    activePayMethod = m;
    openCheckoutSheet(selectedPackIdx);
  }

  function completeDemoPayment(credits, label) {
    S.balance += credits;
    saveAppState();
    syncHeaderAndPrices();
    renderCabinetView();
    closeSheet();
    toast(`✅ +${credits} ★ зачислено (${label})`);
    haptic('success');
  }

  async function executeTelegramStarsPayment(idx) {
    const p = PACKAGES[idx];
    if (!p) return;
    const btn = document.getElementById('starsPayExecBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'СОЗДАЁМ СЧЁТ...'; }
    try {
      const resp = await fetch('/api/create-stars-invoice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pack_id: p.id,
          user_id: (tgUser && tgUser.id) ? tgUser.id : 0,
          lang: 'ru'
        })
      });
      const data = await resp.json();
      if (!resp.ok || !data.invoice_link) throw new Error(data.detail || 'Ошибка создания счёта');

      if (tg && tg.openInvoice) {
        tg.openInvoice(data.invoice_link, function(status) {
          if (status === 'paid') {
            completeDemoPayment(p.credits, 'Telegram Stars');
          } else if (btn) {
            btn.disabled = false;
            btn.textContent = `⭐ ОПЛАТИТЬ ${p.stars} STARS`;
          }
        });
      } else {
        window.open(data.invoice_link, '_blank');
        if (btn) {
          btn.disabled = false;
          btn.textContent = `⭐ ОПЛАТИТЬ ${p.stars} STARS`;
        }
      }
    } catch (e) {
      toast(e.message || 'Ошибка оплаты');
      if (btn) {
        btn.disabled = false;
        btn.textContent = `⭐ ОПЛАТИТЬ ${p.stars} STARS`;
      }
    }
  }

  function redeemPromoCode() {
    const inp = document.getElementById('promoCodeInp');
    const code = (inp.value || '').trim().toUpperCase();
    if (!code) return;
    if (code === 'DEVMODE' || code === 'DEVELOPER') {
      inp.value = '';
      setDevModeState(true);
      return;
    }
    if (S.promoUsed) { toast('Промокод уже был активирован'); haptic('error'); return; }
    const bonus = PROMOS[code];
    if (!bonus) { toast('Промокод не найден'); haptic('error'); return; }
    S.promoUsed = true;
    S.balance += bonus;
    saveAppState();
    syncHeaderAndPrices();
    renderCabinetView();
    inp.value = '';
    toast(`🎁 Промокод активирован: +${bonus} ★`);
    haptic('success');
  }

  // ================= Boot =================
  syncHeaderAndPrices();
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
