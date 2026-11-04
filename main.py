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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo

from google import genai
from google.genai import types as genai_types

# =====================================================================
# SETTINGS
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
APP_URL = os.getenv("RENDER_EXTERNAL_URL", "https://sniper-ugc.onrender.com").strip()
APP_URL_VERSIONED = APP_URL.rstrip("/") + "/?v=4"

# Main model + fallbacks (comma separated). Change in Render -> Environment.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
GEMINI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash").split(",")
    if m.strip()
]

REQUEST_TIMEOUT_MS = 45000      # timeout of one Gemini call
TOTAL_BUDGET_SEC = 95           # total time budget for all retries
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "1").strip() == "1"
MAX_OUTPUT_TOKENS = 4000

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

# ---- Auto-discovery of available Gemini models (survives model retirements) ----
_MODEL_CACHE = {"ts": 0.0, "models": []}
_BAD_WORDS = ("image", "tts", "audio", "live", "embed", "vision", "robotics",
              "computer", "learnlm", "gemma", "aqa", "imagen", "veo")


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
    # stable first, previews last; newest version first; "lite" last
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


class AnalyzeRequest(BaseModel):
    query: str
    format_type: str = "ugc"
    tone: str = "native"
    lang: str = "ru"


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
        "no_model": "Не нашлось доступной модели Gemini. Проверь GEMINI_MODEL в Environment или попробуй позже.",
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
        "no_model": "No available Gemini model found. Check GEMINI_MODEL in Environment or try later.",
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
    """Returns (base_url, card_json) or (default_base_url, None)."""
    path = f"/vol{vol}/part{part}/{nm_id}"
    default_base = get_wb_basket_host(vol) + path

    try:
        r = await client.get(f"{default_base}/info/ru/card.json", timeout=5.0)
        if r.status_code == 200:
            return default_base, r.json()
    except Exception as e:
        logging.warning(f"WB primary basket failed: {e}")

    async def probe(num: int):
        base = f"https://basket-{num:02d}.wbbasket.ru{path}"
        try:
            r = await client.get(f"{base}/info/ru/card.json", timeout=4.0)
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
        "title": f"WB #{nm_id}",
        "brand": "",
        "description": "",
        "features": [],
        "price": "",
        "image": f"{base_url}/images/big/1.webp"
    }

    if card:
        try:
            info["title"] = card.get("imt_name") or info["title"]
            info["brand"] = (card.get("selling") or {}).get("brand_name", "")
            info["description"] = (card.get("description") or "")[:1500]
            opts = card.get("options") or []
            info["features"] = [f"{o.get('name')}: {o.get('value')}" for o in opts[:8]]
        except Exception as e:
            logging.warning(f"WB card parse warning: {e}")

    try:
        card_api = f"https://card.wb.ru/cards/v2/detail?appType=1&curr=rub&dest=-1257786&spp=30&nm={nm_id}"
        r_detail = await client.get(card_api, timeout=6.0)
        if r_detail.status_code == 200:
            prods = r_detail.json().get("data", {}).get("products", [])
            if prods:
                p = prods[0]
                if not info["brand"]:
                    info["brand"] = p.get("brand", "")
                if info["title"].startswith("WB #"):
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8"
    }

    wb_match = re.search(r"wildberries\.ru/catalog/(\d+)", raw)
    num_match = re.fullmatch(r"\d{6,12}", raw)
    if wb_match or num_match:
        nm_id = int(wb_match.group(1) if wb_match else raw)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            return await parse_wildberries_safe(nm_id, client)

    url_match = re.search(r"(https?://[^\s]+)", raw)
    if url_match:
        url = url_match.group(1)
        fallback_data = {
            "source": "Web Link",
            "url": url,
            "title": f"Product: {url}",
            "description": "",
            "features": [],
            "image": "",
            "raw_text": f"Product link: {url}\nUser context: {raw}"
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=8.0) as client:
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
        return fallback_data

    return {
        "source": "Text",
        "title": raw[:80],
        "description": raw[:1500],
        "features": [],
        "image": ""
    }


# =====================================================================
# GEMINI GENERATION (compact output = faster)
# =====================================================================
SYSTEM_INSTRUCTION = """
You are a top producer of viral short-form content for TikTok, Reels and Shorts.
Analyze the product and produce a compact, ready-to-shoot promotion funnel.
You ALWAYS reply with STRICTLY valid JSON, no markdown, no ```json fences.
Keep every text field SHORT: one sentence at most. Speed matters.

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
      "virality_score": 94,
      "hook": "string",
      "scenes": [
        {
          "timecode": "0:00-0:03",
          "visual": "string",
          "voiceover": "string (max 15 words)",
          "on_screen_text": "string (max 6 words)",
          "ai_video_prompt": "English cinematic prompt for Kling/Runway, max 30 words"
        }
      ],
      "cta": "string"
    }
  ],
  "video_generator_prompts": ["English prompt", "English prompt"],
  "seo_caption": "string (2 sentences max)",
  "hashtags": ["#tag", "#tag", "#tag", "#tag", "#tag"]
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


def _parse_model_json(text) -> dict:
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
    if not isinstance(parsed, dict) or not parsed.get("scripts"):
        raise AppError("bad_json")
    return parsed


async def generate_ugc_payload(data: dict, format_type: str, tone: str, lang: str = "ru") -> dict:
    if not ai_client:
        raise AppError("no_key")

    lang = lang if lang in LANG_NAMES else "ru"

    user_prompt = f"""
PRODUCT DATA:
{json.dumps(data, ensure_ascii=False)}

VIDEO FORMAT: {FORMAT_LABELS.get(format_type, format_type)}
TONE: {TONE_LABELS.get(tone, tone)}
OUTPUT LANGUAGE: write ALL text fields in {LANG_NAMES[lang]} (except ai_video_prompt and video_generator_prompts, which are always in English).

REQUIREMENTS: exactly 2 scripts with 3 to 4 scenes each, 5 hooks, 3 pain points, 3 killer features, 2 video generator prompts. Be concise.
"""

    config = genai_types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
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
                parsed = _parse_model_json(getattr(resp, "text", None))
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
            _MODEL_CACHE["ts"] = 0.0  # re-discover models on the next request
        # the real reason matters more than "model gone" from a fallback
        if busy_error is not None:
            raise busy_error
        if bad_answer is not None:
            raise bad_answer
        if gone_error is not None:
            raise gone_error
        raise AppError("busy")

    parsed["meta_source"] = data.get("source", "Web")
    parsed["meta_image"] = data.get("image", "")
    parsed["meta_price"] = data.get("price", "")
    parsed["meta_brand"] = data.get("brand", "")
    return parsed


# Small in-memory cache: same request = instant answer
_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_CACHE_MAX = 64


async def run_pipeline(query: str, format_type: str, tone: str, lang: str) -> dict:
    key = (query.strip().lower(), format_type, tone, lang)
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
# TELEGRAM BOT
# =====================================================================
BOT_TXT = {
    "ru": {
        "start": "<b>SNIPER // UGC GROWTH ENGINE</b>\n\nОтправь артикул или ссылку на Wildberries / любой сайт, чтобы получить готовые сценарии, "
                 "или открой студию по кнопке ниже:",
        "open": "⚡ Открыть SNIPER UGC",
        "wait": "🔍 <b>Анализирую продукт и собираю воронку...</b>",
        "hooks": "🔥 <b>ТОП-ХУКИ:</b>",
        "script": "🎬 <b>СЦЕНАРИЙ:",
        "viral": "виральность",
        "hook": "Хук:",
        "full": "⚡ Открыть полную раскадровку",
        "err": "⚠️ Ошибка:",
    },
    "en": {
        "start": "<b>SNIPER // UGC GROWTH ENGINE</b>\n\nSend a Wildberries link / SKU or any product link to get ready-to-shoot scripts, "
                 "or open the studio with the button below:",
        "open": "⚡ Open SNIPER UGC",
        "wait": "🔍 <b>Analyzing the product and building the funnel...</b>",
        "hooks": "🔥 <b>TOP HOOKS:</b>",
        "script": "🎬 <b>SCRIPT:",
        "viral": "virality",
        "hook": "Hook:",
        "full": "⚡ Open full storyboard",
        "err": "⚠️ Error:",
    },
}


def user_lang(message: types.Message) -> str:
    code = message.from_user.language_code if message.from_user else ""
    return norm_lang(code)


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
# FASTAPI
# =====================================================================
async def keep_alive_loop():
    """Pings our own public URL so the free Render instance does not fall asleep."""
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


FRONTEND_UI = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<title>SNIPER // UGC</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Unbounded:wght@500;600;700&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #07070a;
    --surface: #111116;
    --surface2: #17171e;
    --border: #25252f;
    --text: #f5f5f7;
    --muted: #8c8c9a;
    --accent: #d4ff3a;
    --accent-ink: #0b0b0b;
    --danger: #ff5c5c;
    --ok: #4ade80;
    --radius: 16px;
    --font: 'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
    --display: 'Unbounded', 'Manrope', sans-serif;
    --mono: 'JetBrains Mono', ui-monospace, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
  html, body { background: var(--bg); }
  body {
    color: var(--text);
    font-family: var(--font);
    font-size: 14px;
    line-height: 1.45;
    min-height: 100vh;
    background:
      radial-gradient(600px 300px at 90% -5%, rgba(212,255,58,0.07), transparent 60%),
      radial-gradient(500px 300px at -10% 10%, rgba(120,110,255,0.07), transparent 60%),
      var(--bg);
  }
  .app { max-width: 520px; margin: 0 auto; padding: 14px 16px calc(96px + env(safe-area-inset-bottom)); }
  .mono { font-family: var(--mono); }
  button { font-family: inherit; }

  /* Top bar */
  .topbar { display: flex; justify-content: space-between; align-items: center; gap: 10px; margin-bottom: 18px; }
  .brand { display: flex; align-items: center; gap: 8px; font-family: var(--display); font-weight: 700; font-size: 14px; letter-spacing: 0.5px; }
  .brand .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 12px var(--accent); }
  .brand .sub { color: var(--muted); font-weight: 500; }
  .top-actions { display: flex; align-items: center; gap: 8px; }
  .bal-chip {
    display: flex; align-items: center; gap: 6px;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 7px 11px; border-radius: 999px; font-weight: 700; font-size: 13px; cursor: pointer;
  }
  .bal-chip .coin { color: var(--accent); }
  .lang-sw { display: flex; background: var(--surface); border: 1px solid var(--border); border-radius: 999px; padding: 3px; }
  .lang-sw button {
    background: transparent; border: none; color: var(--muted); font-weight: 700; font-size: 11px;
    padding: 5px 9px; border-radius: 999px; cursor: pointer; letter-spacing: 0.5px;
  }
  .lang-sw button.active { background: var(--text); color: #000; }

  /* Views */
  .view { display: none; }
  .view.active { display: block; animation: fade .25s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

  h1.title { font-family: var(--display); font-size: 20px; font-weight: 600; line-height: 1.25; margin-bottom: 6px; }
  p.lead { color: var(--muted); font-size: 13px; margin-bottom: 16px; }

  .panel, .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px; margin-bottom: 14px;
  }
  .lbl { font-family: var(--mono); font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 8px; display: block; }
  textarea.inp {
    width: 100%; min-height: 80px; background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
    color: var(--text); padding: 12px; font-size: 14px; font-family: inherit; resize: none; outline: none; margin-bottom: 14px;
  }
  textarea.inp:focus, input.inp:focus { border-color: var(--accent); }
  input.inp {
    flex: 1; min-width: 0; background: var(--bg); border: 1px solid var(--border); border-radius: 12px;
    color: var(--text); padding: 11px 12px; font-size: 14px; font-family: inherit; outline: none;
  }
  .chips { display: flex; gap: 6px; overflow-x: auto; margin-bottom: 14px; scrollbar-width: none; }
  .chips::-webkit-scrollbar { display: none; }
  .chip {
    flex-shrink: 0; background: var(--bg); color: var(--muted); border: 1px solid var(--border);
    border-radius: 999px; padding: 8px 13px; font-size: 12px; font-weight: 600; cursor: pointer;
  }
  .chip.active { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }

  .btn {
    width: 100%; background: var(--accent); color: var(--accent-ink); border: none; padding: 14px;
    font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.4px;
    border-radius: 12px; cursor: pointer; display: flex; justify-content: center; align-items: center; gap: 8px;
  }
  .btn:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn.ghost { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
  .btn.small { width: auto; padding: 9px 14px; font-size: 12px; }
  .cost-note { text-align: center; color: var(--muted); font-size: 11px; margin-top: 8px; font-family: var(--mono); }

  .err-box {
    display: none; background: rgba(255,92,92,0.08); border: 1px solid rgba(255,92,92,0.35); color: #ffb3b3;
    border-radius: 12px; padding: 12px; margin-bottom: 14px; font-size: 13px;
  }
  .loader { display: none; padding: 26px 16px; border: 1px dashed var(--border); border-radius: var(--radius); text-align: center; margin-bottom: 14px; }
  .spinner {
    width: 26px; height: 26px; border: 3px solid var(--border); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .8s linear infinite; margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  #results { display: none; }
  .prod-head { display: flex; gap: 12px; align-items: center; padding: 12px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); margin-bottom: 14px; }
  .prod-thumb { width: 54px; height: 54px; border-radius: 12px; object-fit: cover; background: #111; flex-shrink: 0; }
  .tag { font-family: var(--mono); font-size: 10px; border: 1px solid var(--border); padding: 3px 7px; border-radius: 999px; color: var(--muted); }
  .tabs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; background: var(--surface); padding: 4px; border: 1px solid var(--border); border-radius: 14px; margin-bottom: 14px; }
  .tab-btn { background: transparent; color: var(--muted); border: none; padding: 9px 2px; font-size: 10.5px; font-weight: 700; text-transform: uppercase; border-radius: 10px; cursor: pointer; }
  .tab-btn.active { background: var(--text); color: #000; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .card-top { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  .copy-btn { background: transparent; border: 1px solid var(--border); color: var(--muted); padding: 4px 9px; border-radius: 8px; font-size: 10px; font-weight: 700; cursor: pointer; flex-shrink: 0; font-family: var(--mono); }
  .copy-btn:active { background: var(--accent); color: #000; }
  .hook-box { background: var(--bg); border-left: 3px solid var(--accent); padding: 10px 12px; font-size: 13px; margin-bottom: 10px; border-radius: 0 10px 10px 0; }
  .scene-item { background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; font-size: 12.5px; }
  .prompt-box { margin-top: 8px; padding: 8px 10px; background: var(--surface); border: 1px dashed var(--border); border-radius: 10px; font-size: 11px; color: #b0b0bd; display: flex; justify-content: space-between; align-items: center; gap: 8px; }

  /* Cabinet */
  .profile { display: flex; align-items: center; gap: 12px; }
  .avatar { width: 52px; height: 52px; border-radius: 50%; background: var(--surface2); border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; font-family: var(--display); font-size: 18px; font-weight: 700; color: var(--accent); object-fit: cover; flex-shrink: 0; overflow: hidden; }
  .avatar img { width: 100%; height: 100%; object-fit: cover; }
  .pname { font-weight: 800; font-size: 16px; }
  .psub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .balance-card {
    background: linear-gradient(135deg, rgba(212,255,58,0.14), rgba(120,110,255,0.10)), var(--surface);
    border: 1px solid var(--border); border-radius: 20px; padding: 18px; margin-bottom: 14px;
  }
  .bal-num { font-family: var(--display); font-size: 38px; font-weight: 700; line-height: 1.1; margin: 6px 0 2px; }
  .bal-unit { color: var(--muted); font-size: 12px; margin-bottom: 14px; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .stat { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 12px; }
  .stat .v { font-family: var(--display); font-size: 20px; font-weight: 600; }
  .stat .k { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .sec-title { font-family: var(--display); font-size: 13px; font-weight: 600; margin: 18px 0 10px; }
  .packs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 14px; }
  .pack { background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 12px 8px; text-align: center; cursor: pointer; position: relative; color: var(--text); }
  .pack.pop { border-color: var(--accent); }
  .pack .pc { font-family: var(--display); font-size: 20px; font-weight: 600; }
  .pack .pk { color: var(--muted); font-size: 10.5px; margin-bottom: 8px; }
  .pack .pp { font-weight: 800; font-size: 13px; }
  .pack .badge { position: absolute; top: -8px; left: 50%; transform: translateX(-50%); background: var(--accent); color: #000; font-size: 9px; font-weight: 800; padding: 2px 7px; border-radius: 999px; white-space: nowrap; text-transform: uppercase; }
  .row-flex { display: flex; gap: 8px; }
  .list-item { display: flex; justify-content: space-between; align-items: center; gap: 10px; padding: 10px 0; border-bottom: 1px solid var(--border); }
  .list-item:last-child { border-bottom: none; }
  .li-main { min-width: 0; }
  .li-title { font-weight: 700; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .li-sub { color: var(--muted); font-size: 11px; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .plus { color: var(--ok); font-weight: 800; }
  .minus { color: var(--danger); font-weight: 800; }
  .empty { color: var(--muted); font-size: 12px; text-align: center; padding: 14px 0; }
  .demo-note { color: var(--muted); font-size: 11px; text-align: center; margin-top: 8px; }
  .link-btn { background: none; border: none; color: var(--muted); font-size: 11px; cursor: pointer; text-decoration: underline; }

  /* Bottom nav */
  .nav {
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 50;
    padding: 8px 16px calc(8px + env(safe-area-inset-bottom));
    background: rgba(7,7,10,0.88); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border-top: 1px solid var(--border);
  }
  .nav-inner { max-width: 520px; margin: 0 auto; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .nav button {
    background: transparent; border: none; color: var(--muted); display: flex; flex-direction: column; align-items: center; gap: 3px;
    font-size: 11px; font-weight: 700; padding: 6px; border-radius: 12px; cursor: pointer;
  }
  .nav button.active { color: var(--accent); }
  .nav svg { width: 22px; height: 22px; stroke: currentColor; fill: none; stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round; }

  /* Modal */
  .overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.65); backdrop-filter: blur(4px); -webkit-backdrop-filter: blur(4px); z-index: 100; display: none; align-items: flex-end; justify-content: center; }
  .overlay.open { display: flex; }
  .sheet { width: 100%; max-width: 520px; background: var(--surface); border: 1px solid var(--border); border-radius: 22px 22px 0 0; padding: 20px 18px calc(20px + env(safe-area-inset-bottom)); animation: up .22s ease; }
  @keyframes up { from { transform: translateY(30px); opacity: 0; } to { transform: none; opacity: 1; } }
  .sheet h3 { font-family: var(--display); font-size: 16px; font-weight: 600; margin-bottom: 12px; }
  .sheet .kv { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
  .sheet .kv:last-of-type { border-bottom: none; margin-bottom: 8px; }
  .sheet p { color: var(--muted); font-size: 12px; margin: 10px 0 14px; }
  .sheet .btn + .btn { margin-top: 8px; }

  .toast {
    position: fixed; bottom: calc(86px + env(safe-area-inset-bottom)); left: 50%; transform: translateX(-50%);
    background: var(--text); color: #000; padding: 8px 16px; border-radius: 999px; font-size: 12px; font-weight: 800;
    display: none; z-index: 200; white-space: nowrap; max-width: 90vw; overflow: hidden; text-overflow: ellipsis;
  }
</style>
</head>
<body>
<div class="app">

  <div class="topbar">
    <div class="brand"><span class="dot"></span><span>SNIPER</span><span class="sub">UGC</span></div>
    <div class="top-actions">
      <button class="bal-chip" onclick="showView('cabinet')"><span class="coin">◎</span><span id="balChip">0</span></button>
      <div class="lang-sw">
        <button data-l="ru" onclick="setLang('ru')">RU</button>
        <button data-l="en" onclick="setLang('en')">EN</button>
      </div>
    </div>
  </div>

  <!-- STUDIO -->
  <section id="view-studio" class="view active">
    <h1 class="title" data-i18n="studio_title"></h1>
    <p class="lead" data-i18n="studio_lead"></p>

    <div class="panel">
      <span class="lbl" data-i18n="src_label"></span>
      <textarea id="queryInp" class="inp" data-i18n-ph="src_ph"></textarea>

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
        <span data-i18n="generate"></span><span>→</span>
      </button>
      <div class="cost-note" id="costNote"></div>
    </div>

    <div id="errBox" class="err-box"></div>

    <div id="loader" class="loader">
      <div class="spinner"></div>
      <div id="loaderTxt" class="mono" style="font-size:11px; color:var(--muted);"></div>
    </div>

    <div id="results">
      <div class="prod-head" id="pHeader"></div>
      <div class="tabs">
        <button class="tab-btn active" onclick="setTab('scripts', this)" data-i18n="tab_scripts"></button>
        <button class="tab-btn" onclick="setTab('hooks', this)" data-i18n="tab_hooks"></button>
        <button class="tab-btn" onclick="setTab('prompts', this)" data-i18n="tab_video"></button>
        <button class="tab-btn" onclick="setTab('seo', this)" data-i18n="tab_seo"></button>
      </div>
      <div id="tab-scripts" class="tab-content active"></div>
      <div id="tab-hooks" class="tab-content"></div>
      <div id="tab-prompts" class="tab-content"></div>
      <div id="tab-seo" class="tab-content"></div>
    </div>
  </section>

  <!-- CABINET -->
  <section id="view-cabinet" class="view">
    <div class="card"><div class="profile" id="cabProfile"></div></div>

    <div class="balance-card">
      <span class="lbl" data-i18n="balance"></span>
      <div class="bal-num" id="cabBalance">0</div>
      <div class="bal-unit" data-i18n="credits"></div>
      <button class="btn" onclick="scrollToTopup()"><span data-i18n="topup"></span><span>+</span></button>
    </div>

    <div class="stats" id="cabStats"></div>

    <div class="sec-title" id="topupSec" data-i18n="topup_title"></div>
    <div class="packs" id="cabPacks"></div>

    <div class="card">
      <span class="lbl" data-i18n="promo"></span>
      <div class="row-flex">
        <input id="promoInp" class="inp" data-i18n-ph="promo_ph" autocapitalize="characters" autocomplete="off">
        <button class="btn small" onclick="applyPromo()" data-i18n="apply"></button>
      </div>
    </div>
    <div class="demo-note" data-i18n="demo_note"></div>

    <div class="sec-title" style="display:flex; justify-content:space-between; align-items:center;">
      <span data-i18n="history"></span>
      <button class="link-btn" onclick="clearHistory()" data-i18n="clear"></button>
    </div>
    <div class="card" id="cabHistory"></div>

    <div class="sec-title" data-i18n="transactions"></div>
    <div class="card" id="cabTx"></div>
  </section>

</div>

<!-- Bottom nav -->
<nav class="nav">
  <div class="nav-inner">
    <button id="nav-studio" class="active" onclick="showView('studio')">
      <svg viewBox="0 0 24 24"><path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3z"/><path d="M19 17l.8 2.2L22 20l-2.2.8L19 23l-.8-2.2L16 20l2.2-.8L19 17z"/></svg>
      <span data-i18n="nav_studio"></span>
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
  // ================= Telegram =================
  const tg = (window.Telegram && window.Telegram.WebApp) ? window.Telegram.WebApp : null;
  if (tg) {
    try {
      tg.ready();
      tg.expand();
      if (tg.setHeaderColor) tg.setHeaderColor('#07070a');
      if (tg.setBackgroundColor) tg.setBackgroundColor('#07070a');
    } catch (e) {}
  }
  const tgUser = (tg && tg.initDataUnsafe && tg.initDataUnsafe.user) ? tg.initDataUnsafe.user : null;

  // ================= i18n =================
  const I18N = {
    ru: {
      nav_studio: 'Студия', nav_cabinet: 'Кабинет',
      studio_title: 'Виральные сценарии за секунды',
      studio_lead: 'Вставь ссылку или артикул — получи хуки, раскадровку и промпты для ИИ-видео.',
      src_label: 'Источник (ссылка / артикул / описание)',
      src_ph: 'Ссылка на Wildberries, Ozon, любой сайт, артикул или описание товара...',
      fmt_label: 'Формат ролика', tone_label: 'Тональность',
      f_ugc: 'UGC обзор', f_ps: 'Боль → Решение', f_unb: 'Распаковка / ASMR', f_aes: 'Aesthetic', f_story: 'Сторителлинг',
      t_native: 'Нативный', t_aggr: 'Дерзкий', t_humor: 'Юмор', t_lux: 'Премиум',
      generate: 'Сгенерировать', cost: 'Стоимость: {n} кр.',
      ld1: 'ИЗВЛЕКАЮ ДАННЫЕ О ТОВАРЕ...', ld2: 'ПРИДУМЫВАЮ ХУКИ...', ld3: 'ПИШУ РАСКАДРОВКУ...', ld4: 'ПОЧТИ ГОТОВО, ЕЩЁ НЕМНОГО...',
      tab_scripts: 'Сценарии', tab_hooks: 'Хуки', tab_video: 'AI Видео', tab_seo: 'ЦА & SEO',
      frame: 'Кадр', voice: 'Голос', screen_text: 'Текст', hook: 'ХУК', cta: 'CTA',
      copy: 'COPY', copy_all: 'ВЕСЬ СЦЕНАРИЙ', copy_seo: 'COPY SEO', copied: 'Скопировано',
      audience: 'Целевая аудитория', pains: 'Боли ЦА', features: 'Главные фишки', seo_title: 'SEO описание и теги',
      virality: 'виральность', script: 'Сценарий',
      balance: 'Баланс', credits: 'кредитов', topup: 'Пополнить', topup_title: 'Пополнение баланса',
      gens: 'Генераций', spent: 'Потрачено кр.',
      pack_credits: 'кредитов', popular: 'Хит',
      promo: 'Промокод', promo_ph: 'Введи промокод', apply: 'Применить',
      promo_ok: 'Промокод активирован: +{n} кр.', promo_bad: 'Неверный промокод', promo_used: 'Промокод уже использован',
      demo_note: 'Демо-режим: оплата не списывает реальные деньги.',
      history: 'История генераций', clear: 'Очистить', history_empty: 'Пока пусто. Сгенерируй первый сценарий.',
      transactions: 'Операции', tx_empty: 'Операций пока нет',
      tx_bonus: 'Приветственный бонус', tx_topup: 'Пополнение', tx_gen: 'Генерация сценариев', tx_promo: 'Промокод',
      open: 'Открыть', del: 'Удалить',
      plan: 'Демо-тариф', guest: 'Гость',
      pay_title: 'Демо-оплата', pay_pack: 'Пакет', pay_price: 'Стоимость',
      pay_note: 'Это демо: деньги не списываются, кредиты начислятся сразу.',
      pay_confirm: 'Оплатить (демо)', cancel: 'Отмена', processing: 'Обработка...',
      pay_done: 'Баланс пополнен: +{n} кр.',
      nofunds_title: 'Недостаточно кредитов',
      nofunds_text: 'Для генерации нужно {n} кр. Пополни баланс, чтобы продолжить.',
      err_query: 'Укажи ссылку, артикул или описание товара',
      err_timeout: 'Сервер отвечает слишком долго. Попробуй ещё раз.',
      err_network: 'Не удалось связаться с сервером. Проверь интернет.',
      err_generic: 'Не удалось обработать запрос',
      cleared: 'История очищена'
    },
    en: {
      nav_studio: 'Studio', nav_cabinet: 'Account',
      studio_title: 'Viral scripts in seconds',
      studio_lead: 'Paste a link or SKU to get hooks, storyboards and AI video prompts.',
      src_label: 'Source (link / SKU / description)',
      src_ph: 'Wildberries, Ozon or any website link, SKU, or product description...',
      fmt_label: 'Video format', tone_label: 'Tone',
      f_ugc: 'UGC review', f_ps: 'Pain → Solution', f_unb: 'Unboxing / ASMR', f_aes: 'Aesthetic', f_story: 'Storytelling',
      t_native: 'Native', t_aggr: 'Bold', t_humor: 'Humor', t_lux: 'Premium',
      generate: 'Generate', cost: 'Cost: {n} cr.',
      ld1: 'EXTRACTING PRODUCT DATA...', ld2: 'CRAFTING HOOKS...', ld3: 'WRITING STORYBOARD...', ld4: 'ALMOST DONE, HANG ON...',
      tab_scripts: 'Scripts', tab_hooks: 'Hooks', tab_video: 'AI Video', tab_seo: 'Audience & SEO',
      frame: 'Frame', voice: 'Voice', screen_text: 'Text', hook: 'HOOK', cta: 'CTA',
      copy: 'COPY', copy_all: 'FULL SCRIPT', copy_seo: 'COPY SEO', copied: 'Copied',
      audience: 'Target audience', pains: 'Audience pains', features: 'Killer features', seo_title: 'SEO caption & tags',
      virality: 'virality', script: 'Script',
      balance: 'Balance', credits: 'credits', topup: 'Top up', topup_title: 'Top up balance',
      gens: 'Generations', spent: 'Credits spent',
      pack_credits: 'credits', popular: 'Hot',
      promo: 'Promo code', promo_ph: 'Enter promo code', apply: 'Apply',
      promo_ok: 'Promo applied: +{n} cr.', promo_bad: 'Invalid promo code', promo_used: 'Promo code already used',
      demo_note: 'Demo mode: payment does not charge real money.',
      history: 'Generation history', clear: 'Clear', history_empty: 'Nothing here yet. Generate your first script.',
      transactions: 'Transactions', tx_empty: 'No transactions yet',
      tx_bonus: 'Welcome bonus', tx_topup: 'Top up', tx_gen: 'Script generation', tx_promo: 'Promo code',
      open: 'Open', del: 'Delete',
      plan: 'Demo plan', guest: 'Guest',
      pay_title: 'Demo payment', pay_pack: 'Pack', pay_price: 'Price',
      pay_note: 'This is a demo: no money is charged, credits are added instantly.',
      pay_confirm: 'Pay (demo)', cancel: 'Cancel', processing: 'Processing...',
      pay_done: 'Balance topped up: +{n} cr.',
      nofunds_title: 'Not enough credits',
      nofunds_text: 'Generation costs {n} cr. Top up your balance to continue.',
      err_query: 'Enter a link, SKU or product description',
      err_timeout: 'The server is taking too long. Please try again.',
      err_network: 'Could not reach the server. Check your connection.',
      err_generic: 'Could not process the request',
      cleared: 'History cleared'
    }
  };

  // ================= State (demo, stored in the browser) =================
  const STORE_KEY = 'sniper_state_v3';
  const COST = 10;
  const WELCOME_BONUS = 100;
  const PROMO_CODES = { 'SNIPER': 50 };
  const PACKS = [
    { credits: 50,  ru: '490 ₽',  en: '$4.90',  pop: false },
    { credits: 150, ru: '1 190 ₽', en: '$11.90', pop: true },
    { credits: 500, ru: '2 990 ₽', en: '$29.90', pop: false }
  ];

  function detectLang() {
    const c = ((tgUser && tgUser.language_code) || navigator.language || 'en').slice(0, 2).toLowerCase();
    return ['ru', 'uk', 'be', 'kk'].indexOf(c) >= 0 ? 'ru' : 'en';
  }
  function defaultState() {
    return {
      lang: detectLang(),
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
        if (o && typeof o === 'object') return Object.assign(defaultState(), o);
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
  if (!I18N[S.lang]) S.lang = 'en';

  function t(key, vars) {
    let s = (I18N[S.lang] && I18N[S.lang][key]) || I18N.en[key] || key;
    if (vars) Object.keys(vars).forEach(k => { s = s.split('{' + k + '}').join(vars[k]); });
    return s;
  }

  // ================= Helpers =================
  let curFmt = 'ugc';
  let curTone = 'native';
  let lastResult = null;
  const copyStore = [];

  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }
  function reg(txt) { copyStore.push(txt == null ? '' : String(txt)); return copyStore.length - 1; }
  function copyIdx(i) { copy(copyStore[i]); }

  let toastTimer = null;
  function showToast(text) {
    const el = document.getElementById('toast');
    el.textContent = text;
    el.style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.style.display = 'none'; }, 1800);
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

  // ================= Language =================
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
    document.getElementById('costNote').textContent = t('cost', { n: COST });
    renderCabinet();
    if (lastResult) renderResult(lastResult);
  }

  // ================= Views / tabs / chips =================
  function showView(name) {
    ['studio', 'cabinet'].forEach(v => {
      document.getElementById('view-' + v).classList.toggle('active', v === name);
      document.getElementById('nav-' + v).classList.toggle('active', v === name);
    });
    if (name === 'cabinet') renderCabinet();
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

  // ================= Modal =================
  function openModal(html) {
    document.getElementById('sheet').innerHTML = html;
    document.getElementById('overlay').classList.add('open');
  }
  function closeModal() {
    document.getElementById('overlay').classList.remove('open');
  }

  // ================= Balance =================
  function updateBalanceUI() {
    document.getElementById('balChip').textContent = S.balance;
    const cb = document.getElementById('cabBalance');
    if (cb) cb.textContent = S.balance;
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
      <div class="kv"><span>${esc(t('pay_pack'))}</span><strong>${p.credits} ${esc(t('pack_credits'))}</strong></div>
      <div class="kv"><span>${esc(t('pay_price'))}</span><strong>${esc(p[S.lang])}</strong></div>
      <p>${esc(t('pay_note'))}</p>
      <button class="btn" id="payBtn" onclick="doPay(${i})">${esc(t('pay_confirm'))}</button>
      <button class="btn ghost" onclick="closeModal()">${esc(t('cancel'))}</button>
    `);
  }
  function doPay(i) {
    const p = PACKS[i];
    if (!p) return;
    const btn = document.getElementById('payBtn');
    if (btn) { btn.disabled = true; btn.textContent = t('processing'); }
    setTimeout(() => {
      S.balance += p.credits;
      addTx('topup', p.credits);
      saveState();
      updateBalanceUI();
      renderCabinet();
      closeModal();
      showToast(t('pay_done', { n: p.credits }));
      haptic('success');
    }, 900);
  }
  function openNoFunds() {
    openModal(`
      <h3>${esc(t('nofunds_title'))}</h3>
      <p>${esc(t('nofunds_text', { n: COST }))}</p>
      <button class="btn" onclick="closeModal(); showView('cabinet'); setTimeout(scrollToTopup, 80);">${esc(t('topup'))}</button>
      <button class="btn ghost" onclick="closeModal()">${esc(t('cancel'))}</button>
    `);
  }
  function applyPromo() {
    const inp = document.getElementById('promoInp');
    const code = (inp.value || '').trim().toUpperCase();
    if (!code) return;
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

  // ================= Cabinet =================
  function renderCabinet() {
    const name = tgUser ? ([tgUser.first_name, tgUser.last_name].filter(Boolean).join(' ') || tgUser.username || t('guest')) : t('guest');
    const sub = tgUser ? (tgUser.username ? '@' + tgUser.username : 'ID ' + tgUser.id) : 'Demo';
    const initial = (name || '?').trim().charAt(0).toUpperCase() || '?';
    const avatar = (tgUser && tgUser.photo_url)
      ? `<div class="avatar"><img src="${esc(tgUser.photo_url)}" alt="" onerror="this.parentNode.textContent='${esc(initial)}'"></div>`
      : `<div class="avatar">${esc(initial)}</div>`;
    document.getElementById('cabProfile').innerHTML = `
      ${avatar}
      <div style="min-width:0;">
        <div class="pname">${esc(name)}</div>
        <div class="psub">${esc(sub)} · <span class="tag">${esc(t('plan'))}</span></div>
      </div>`;

    updateBalanceUI();

    document.getElementById('cabStats').innerHTML = `
      <div class="stat"><div class="v">${S.gens}</div><div class="k">${esc(t('gens'))}</div></div>
      <div class="stat"><div class="v">${S.spent}</div><div class="k">${esc(t('spent'))}</div></div>`;

    document.getElementById('cabPacks').innerHTML = PACKS.map((p, i) => `
      <button class="pack ${p.pop ? 'pop' : ''}" onclick="openPay(${i})">
        ${p.pop ? `<span class="badge">${esc(t('popular'))}</span>` : ''}
        <div class="pc">${p.credits}</div>
        <div class="pk">${esc(t('pack_credits'))}</div>
        <div class="pp">${esc(p[S.lang])}</div>
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

  // ================= Generation =================
  function showErr(msg) {
    const b = document.getElementById('errBox');
    b.textContent = msg;
    b.style.display = 'block';
    haptic('error');
  }
  function hideErr() { document.getElementById('errBox').style.display = 'none'; }

  async function postAnalyze(body, signal) {
    const doFetch = () => fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: signal
    });
    try {
      return await doFetch();
    } catch (e) {
      if (e && e.name === 'AbortError') throw e;
      await new Promise(r => setTimeout(r, 4000)); // server may be waking up
      return await doFetch();
    }
  }

  async function startGeneration() {
    hideErr();
    const q = document.getElementById('queryInp').value.trim();
    if (!q) { showErr(t('err_query')); return; }
    if (S.balance < COST) { openNoFunds(); return; }

    const btn = document.getElementById('submitBtn');
    const ldr = document.getElementById('loader');
    const ldrTxt = document.getElementById('loaderTxt');
    const res = document.getElementById('results');

    btn.disabled = true;
    res.style.display = 'none';
    ldr.style.display = 'block';

    const steps = ['ld1', 'ld2', 'ld3', 'ld4'];
    let si = 0;
    ldrTxt.textContent = t(steps[0]);
    const stepTimer = setInterval(() => {
      if (si < steps.length - 1) { si++; ldrTxt.textContent = t(steps[si]); }
    }, 3500);

    const ctrl = new AbortController();
    const abortTimer = setTimeout(() => ctrl.abort(), 125000);

    try {
      const resp = await postAnalyze({ query: q, format_type: curFmt, tone: curTone, lang: S.lang }, ctrl.signal);

      let payload = {};
      try { payload = await resp.json(); } catch (e) {}
      if (!resp.ok) throw new Error(payload.detail || (resp.status >= 502 ? t('err_network') : t('err_generic')));

      // Success: charge credits and save to history
      S.balance -= COST;
      S.spent += COST;
      S.gens += 1;
      addTx('gen', -COST);
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
      haptic('success');
    } catch (e) {
      if (e && e.name === 'AbortError') showErr(t('err_timeout'));
      else if (e && /Failed to fetch|Load failed|NetworkError/i.test(e.message || '')) showErr(t('err_network'));
      else showErr((e && e.message) || t('err_generic'));
    } finally {
      clearInterval(stepTimer);
      clearTimeout(abortTimer);
      ldr.style.display = 'none';
      btn.disabled = false;
    }
  }

  // ================= Render result =================
  function renderResult(d) {
    copyStore.length = 0;

    const imgHtml = d.meta_image ? `<img class="prod-thumb" src="${esc(d.meta_image)}" alt="" onerror="this.style.display='none'">` : '';
    const price = d.meta_price ? `<span class="tag">${esc(d.meta_price)}</span>` : '';
    document.getElementById('pHeader').innerHTML = `
      ${imgHtml}
      <div style="min-width:0;">
        <div style="font-weight:800; font-size:14px; margin-bottom:6px;">${esc(d.product_title)}</div>
        <div style="display:flex; gap:6px; flex-wrap:wrap;">
          <span class="tag">${esc(d.meta_source || 'WEB')}</span>
          <span class="tag">${esc(d.category || '')}</span>
          ${price}
        </div>
      </div>`;

    // Scripts
    document.getElementById('tab-scripts').innerHTML = (d.scripts || []).map((sc, i) => {
      const full = `${t('script')}: ${sc.title}\n${t('hook')}: ${sc.hook}\n` +
        (sc.scenes || []).map(s => `[${s.timecode}] ${s.visual} | ${t('voice')}: ${s.voiceover}`).join('\n') +
        `\nCTA: ${sc.cta}`;
      const scenes = (sc.scenes || []).map(s => `
        <div class="scene-item">
          <div class="mono" style="color:var(--muted); font-size:10px; margin-bottom:4px;">${esc(s.timecode)}</div>
          <div><strong>${esc(t('frame'))}:</strong> ${esc(s.visual)}</div>
          <div style="margin:3px 0;"><strong>${esc(t('voice'))}:</strong> «${esc(s.voiceover)}»</div>
          <div><strong>${esc(t('screen_text'))}:</strong> ${esc(s.on_screen_text)}</div>
          <div class="prompt-box mono">
            <span>AI: ${esc(s.ai_video_prompt)}</span>
            <button class="copy-btn" onclick="copyIdx(${reg(s.ai_video_prompt)})">${esc(t('copy'))}</button>
          </div>
        </div>`).join('');

      return `
        <div class="card">
          <div class="card-top">
            <div style="min-width:0;">
              <div style="font-weight:800; text-transform:uppercase;">0${i + 1} // ${esc(sc.title)}</div>
              <div class="mono" style="font-size:10px; color:var(--muted);">${esc(sc.angle)} · ${esc(sc.virality_score)}% ${esc(t('virality'))}</div>
            </div>
            <button class="copy-btn" onclick="copyIdx(${reg(full)})">${esc(t('copy_all'))}</button>
          </div>
          <div class="hook-box"><strong>${esc(t('hook'))}:</strong> «${esc(sc.hook)}»</div>
          ${scenes}
          <div style="font-size:12.5px; margin-top:8px;"><strong>CTA:</strong> ${esc(sc.cta)}</div>
        </div>`;
    }).join('');

    // Hooks
    document.getElementById('tab-hooks').innerHTML = `
      <div class="card">
        ${(d.viral_hooks || []).map((hk, i) => `
          <div class="list-item">
            <div style="font-size:13px;"><strong>#0${i + 1}</strong> «${esc(hk)}»</div>
            <button class="copy-btn" onclick="copyIdx(${reg(hk)})">${esc(t('copy'))}</button>
          </div>`).join('')}
      </div>`;

    // AI video prompts
    document.getElementById('tab-prompts').innerHTML = `
      <div class="card">
        ${(d.video_generator_prompts || []).map(p => `
          <div class="prompt-box mono" style="margin-bottom:8px;">
            <span>${esc(p)}</span>
            <button class="copy-btn" onclick="copyIdx(${reg(p)})">${esc(t('copy'))}</button>
          </div>`).join('')}
      </div>`;

    // Audience & SEO
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
  fetch('/health').catch(() => {}); // wake the server up early
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    return HTMLResponse(
        content=FRONTEND_UI,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
