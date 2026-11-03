import os
import re
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

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
# КОНФИГУРАЦИЯ И ИНИЦИАЛИЗАЦИЯ
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
APP_URL = os.getenv("RENDER_EXTERNAL_URL", "https://sniper-ugc.onrender.com")

bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
dp = Dispatcher()
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# =====================================================================
# МОДЕЛИ ДАННЫХ ДЛЯ ГЕНЕРАЦИИ И API
# =====================================================================
class AnalyzeRequest(BaseModel):
    query: str
    format_type: str = "ugc"       # ugc, unboxing, problem_solution, aesthetic, storytelling
    tone: str = "native"           # native, aggressive, humor, luxury

class SceneItem(BaseModel):
    timecode: str
    visual: str
    voiceover: str
    on_screen_text: str
    ai_video_prompt: str

class ScriptVariant(BaseModel):
    title: str
    angle: str
    virality_score: int
    hook: str
    scenes: List[SceneItem]
    cta: str

class UGCResponseSchema(BaseModel):
    product_title: str
    category: str
    target_audience: str
    pain_points: List[str]
    killer_features: List[str]
    viral_hooks: List[str]
    scripts: List[ScriptVariant]
    video_generator_prompts: List[str]
    seo_caption: str
    hashtags: List[str]

# =====================================================================
# УНИВЕРСАЛЬНЫЙ ПАРСЕР (WILDBERRIES, OZON, ЛЮБЫЕ САЙТЫ И ТЕКСТ)
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


async def parse_wildberries(nm_id: int, client: httpx.AsyncClient) -> dict:
    vol = nm_id // 100000
    part = nm_id // 1000
    host = get_wb_basket_host(vol)
    base_url = f"{host}/vol{vol}/part{part}/{nm_id}"

    info = {
        "source": "Wildberries",
        "id": str(nm_id),
        "title": f"Товар Wildberries #{nm_id}",
        "brand": "",
        "description": "",
        "features": [],
        "price": "",
        "image": f"{base_url}/images/big/1.webp"
    }

    # 1. Запрос к внутреннему JSON карточки на CDN-корзине
    try:
        r_card = await client.get(f"{base_url}/info/ru/card.json", timeout=8.0)
        if r_card.status_code == 200:
            data = r_card.json()
            info["title"] = data.get("imt_name") or info["title"]
            info["brand"] = data.get("selling", {}).get("brand_name", "")
            info["description"] = data.get("description", "")
            options = data.get("options", [])
            info["features"] = [f"{opt.get('name')}: {opt.get('value')}" for opt in options[:12]]
    except Exception as e:
        logging.warning(f"WB Basket JSON error: {e}")

    # 2. Запрос к API каталога для цены и рейтинга
    try:
        card_api = f"https://card.wb.ru/cards/v2/detail?appType=1&curr=rub&dest=-1257786&spp=30&nm={nm_id}"
        r_detail = await client.get(card_api, timeout=8.0)
        if r_detail.status_code == 200:
            products = r_detail.json().get("data", {}).get("products", [])
            if products:
                p = products[0]
                if not info["title"] or "Товар Wildberries" in info["title"]:
                    info["title"] = p.get("name", info["title"])
                info["brand"] = info["brand"] or p.get("brand", "")
                sizes = p.get("sizes", [])
                if sizes and "price" in sizes[0]:
                    price_rub = sizes[0]["price"].get("product", 0) // 100
                    if price_rub:
                        info["price"] = f"{price_rub} ₽"
    except Exception as e:
        logging.warning(f"WB Detail API error: {e}")

    return info


async def extract_universal_context(raw_input: str) -> dict:
    raw_input = raw_input.strip()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"
    }

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, verify=False) as client:
        # Проверка на артикул или ссылку Wildberries
        wb_match = re.search(r"wildberries\.ru/catalog/(\d+)", raw_input)
        only_digits = re.fullmatch(r"\d{6,11}", raw_input)

        if wb_match or only_digits:
            nm_id = int(wb_match.group(1) if wb_match else raw_input)
            return await parse_wildberries(nm_id, client)

        # Проверка на URL любого другого сайта (Ozon, Яндекс Маркет, Shopify, лендинг, Telegram и т.д.)
        url_match = re.search(r"(https?://[^\s]+)", raw_input)
        if url_match:
            url = url_match.group(1)
            info = {
                "source": "Web URL",
                "url": url,
                "title": "",
                "description": "",
                "features": [],
                "image": "",
                "raw_excerpt": ""
            }
            try:
                resp = await client.get(url, timeout=12.0)
                soup = BeautifulSoup(resp.text, "html.parser")

                # Вытаскиваем OpenGraph и мета-теги
                og_title = soup.find("meta", property="og:title")
                og_desc = soup.find("meta", property="og:description")
                og_img = soup.find("meta", property="og:image")
                meta_desc = soup.find("meta", attrs={"name": "description"})

                info["title"] = (
                    (og_title["content"] if og_title and og_title.get("content") else "")
                    or (soup.title.string.strip() if soup.title and soup.title.string else "")
                )
                info["description"] = (
                    (og_desc["content"] if og_desc and og_desc.get("content") else "")
                    or (meta_desc["content"] if meta_desc and meta_desc.get("content") else "")
                )
                info["image"] = og_img["content"] if og_img and og_img.get("content") else ""

                # Вытаскиваем JSON-LD микроразметку продуктов, если есть
                for script_tag in soup.find_all("script", type="application/ld+json"):
                    try:
                        ld_data = json.loads(script_tag.string or "{}")
                        if isinstance(ld_data, dict):
                            info["title"] = info["title"] or ld_data.get("name", "")
                            info["description"] = info["description"] or ld_data.get("description", "")
                    except Exception:
                        pass

                # Чистим видимый текст страницы
                for junk in soup(["script", "style", "noscript", "svg", "footer", "nav"]):
                    junk.decompose()
                clean_text = " ".join(soup.stripped_strings)
                info["raw_excerpt"] = clean_text[:3500]
            except Exception as e:
                logging.warning(f"Universal URL scrape fallback ({url}): {e}")
                info["raw_excerpt"] = f"Ссылка пользователя: {url}. Доп. контекст из запроса: {raw_input}"

            return info

        # Если передан просто текст / описание ниши / идея
        return {
            "source": "Прямой запрос",
            "title": raw_input[:90],
            "description": raw_input,
            "features": [],
            "image": ""
        }


# =====================================================================
# ГЕНЕРАЦИЯ ВИРАЛЬНОЙ СТРАТЕГИИ И СЦЕНАРИЕВ (GEMINI API)
# =====================================================================
async def generate_ugc_pack(parsed_data: dict, format_type: str, tone: str) -> dict:
    if not ai_client:
        raise HTTPException(status_code=500, detail="Не задан GEMINI_API_KEY в переменных окружения Render.")

    format_names = {
        "ugc": "Нативный UGC-обзор от лица реального покупателя",
        "unboxing": "ASMR / Быстрая распаковка + первая реакция",
        "problem_solution": "Жесткая боль -> Решение через продукт",
        "aesthetic": "Визуальный минимализм / Aesthetic TikTok / Reels",
        "storytelling": "Личная мини-история с неожиданным твистом"
    }
    tone_names = {
        "native": "Искренний, живой, разговорный ('делюсь находкой с другом')",
        "aggressive": "Дерзкий, триггерный, максимальное удержание и провокация",
        "humor": "Ироничный, с мемами и легким сарказмом",
        "luxury": "Премиальный, лаконичный, фокус на статусе и эстетике"
    }

    prompt = f"""
Ты — топовый креативный продюсер TikTok, Instagram Reels и YouTube Shorts.
Твоя задача — разработать полную виральную воронку и сценарии продвижения на основе входных данных о продукте/сервисе/ссылке.

ВХОДНЫЕ ДАННЫЕ О ПРОДУКТЕ / ССЫЛКЕ:
{json.dumps(parsed_data, ensure_ascii=False, indent=2)}

ВЫБРАННЫЙ ФОРМАТ: {format_names.get(format_type, format_type)}
ТОНАЛЬНОСТЬ: {tone_names.get(tone, tone)}

ТРЕБОВАНИЯ К РЕЗУЛЬТАТУ:
1. Никакой воды и штампов ("В современном мире...", "Привет, друзья"). Только живой язык коротких роликов, удерживающий внимание с первой секунды.
2. Сгенерируй 5 бронебойных хуков (первые 2 секунды видео), которые останавливают скролл ленты.
3. Разработай 3 полноценных разноплановых сценария (с разными углами захода: например, через боль, через скепсис/проверку, через вау-эффект).
4. В каждом сценарии распиши посекундную раскадровку:
   - Таймкод (например, 0:00-0:02)
   - Что конкретно в кадре (Visual)
   - Озвучка (Voiceover — короткие динамичные фразы)
   - Текст на экране (On-screen text)
   - Промпт на английском языке для ИИ-видеогенератора (Kling AI / Runway Gen-3 / Sora) для генерации этого кадра (ai_video_prompt).
5. Добавь 3 детальных мастер-промпта на английском языке для генерации полноценного B-roll видео с этим продуктом в нейросетях (Kling / Luma / Runway).
6. Напиши продающее SEO-описание под ролик и подбери релевантные хэштеги.
"""

    response = await asyncio.to_thread(
        ai_client.models.generate_content,
        model="gemini-2.5-flash",
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=UGCResponseSchema,
            temperature=0.75,
        ),
    )

    result = json.loads(response.text)
    result["meta_source"] = parsed_data.get("source", "Web")
    result["meta_image"] = parsed_data.get("image", "")
    result["meta_price"] = parsed_data.get("price", "")
    result["meta_brand"] = parsed_data.get("brand", "")
    return result


# =====================================================================
# TELEGRAM-БОТ (ОБРАБОТКА ССЫЛОК В ЧАТЕ + КНОПКА MINI APP)
# =====================================================================
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚡ Открыть студию SNIPER UGC", web_app=WebAppInfo(url=APP_URL))]
    ])
    await message.answer(
        "<b>SNIPER UGC // AI GROWTH ENGINE</b>\n\n"
        "Отправь любую ссылку (Wildberries, Ozon, сайт, Telegram-канал) или открой студию по кнопке ниже, "
        "чтобы сгенерировать хуки, раскадровки и промпты для съёмки роликов.",
        reply_markup=kb,
        parse_mode="HTML"
    )

@dp.message()
async def handle_direct_message(message: types.Message):
    text = (message.text or "").strip()
    if not text:
        return

    status_msg = await message.answer("⏳ <b>Сканирую источник и собираю сценарии...</b>", parse_mode="HTML")
    try:
        parsed = await extract_universal_context(text)
        pack = await generate_ugc_pack(parsed, format_type="ugc", tone="native")

        hooks_text = "\n".join([f"— {h}" for h in pack["viral_hooks"][:3]])
        s1 = pack["scripts"][0]
        scenes_text = "\n".join([
            f"<code>[{sc['timecode']}]</code> <b>Кадр:</b> {sc['visual']}\n🗣 <i>«{sc['voiceover']}»</i>"
            for sc in s1["scenes"]
        ])

        reply = (
            f"🎯 <b>{pack['product_title']}</b>\n"
            f"ЦА: {pack['target_audience']}\n\n"
            f"🔥 <b>ТОП-ХУКИ:</b>\n{hooks_text}\n\n"
            f"🎬 <b>СЦЕНАРИЙ #1: {s1['title']} ({s1['virality_score']}% виральность)</b>\n"
            f"{scenes_text}\n\n"
            f"📢 <b>CTA:</b> {s1['cta']}"
        )

        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Открыть полный разбор в Mini App", web_app=WebAppInfo(url=APP_URL))]
        ])
        await status_msg.edit_text(reply[:4000], reply_markup=kb, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Bot error: {e}")
        await status_msg.edit_text(f"⚠️ Ошибка при разборе: {e}")


# =====================================================================
# ЖИЗНЕННЫЙ ЦИКЛ FASTAPI + ЗАПУСК POLLING БОТА
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    polling_task = None
    if bot:
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            polling_task = asyncio.create_task(dp.start_polling(bot, handle_signals=False))
            logging.info("Telegram Bot Polling запущен.")
        except Exception as e:
            logging.error(f"Не удалось запустить бота: {e}")
    yield
    if polling_task:
        polling_task.cancel()
    if bot:
        await bot.session.close()

app = FastAPI(title="SNIPER UGC Engine", lifespan=lifespan)

@app.head("/")
async def healthcheck_head():
    return HTMLResponse(content="", status_code=200)

@app.post("/api/analyze")
async def api_analyze(req: AnalyzeRequest):
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Введите ссылку, артикул или описание продукта.")
    parsed = await extract_universal_context(req.query)
    result = await generate_ugc_pack(parsed, req.format_type, req.tone)
    return result

# =====================================================================
# МИНИМАЛИСТИЧНЫЙ ЧЕРНО-БЕЛЫЙ ИНТЕРФЕЙС (HTML / CSS / JS)
# =====================================================================
FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>SNIPER // UGC</title>
<script src="https://telegram.org/js/telegram-web-app.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #000000;
    --surface: #0a0a0a;
    --surface-hover: #121212;
    --border: #222222;
    --border-active: #ffffff;
    --text: #ffffff;
    --text-muted: #888888;
    --accent: #ffffff;
    --accent-fg: #000000;
    --radius: 8px;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    -webkit-tap-highlight-color: transparent;
  }

  body {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    min-height: 100vh;
    padding: 16px 16px 60px;
    overflow-x: hidden;
  }

  .mono {
    font-family: 'JetBrains Mono', monospace;
  }

  /* HEADER */
  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 16px;
    margin-bottom: 20px;
    border-bottom: 1px solid var(--border);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: -0.03em;
  }

  .brand-dot {
    width: 8px;
    height: 8px;
    background: var(--text);
    border-radius: 50%;
    box-shadow: 0 0 10px #fff;
  }

  .badge-status {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 4px 8px;
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-muted);
  }

  /* INPUT PANEL */
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin-bottom: 16px;
  }

  .label {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .input-wrap {
    position: relative;
    margin-bottom: 14px;
  }

  textarea.main-input {
    width: 100%;
    min-height: 76px;
    background: #050505;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: inherit;
    font-size: 14px;
    padding: 12px;
    resize: none;
    outline: none;
    transition: border-color 0.15s ease;
  }

  textarea.main-input:focus {
    border-color: var(--border-active);
  }

  textarea.main-input::placeholder {
    color: #555555;
  }

  /* CHIPS */
  .chips-scroll {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    padding-bottom: 4px;
    margin-bottom: 14px;
    scrollbar-width: none;
  }
  .chips-scroll::-webkit-scrollbar {
    display: none;
  }

  .chip {
    flex-shrink: 0;
    background: #050505;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .chip.active {
    background: var(--text);
    color: var(--bg);
    border-color: var(--text);
    font-weight: 600;
  }

  /* CTA BUTTON */
  .btn-primary {
    width: 100%;
    background: var(--text);
    color: var(--bg);
    border: 1px solid var(--text);
    border-radius: 6px;
    padding: 13px 16px;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: opacity 0.15s ease, transform 0.08s ease;
  }

  .btn-primary:active {
    transform: scale(0.99);
  }

  .btn-primary:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* LOADER */
  .loader-box {
    display: none;
    padding: 28px 16px;
    text-align: center;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
  }

  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border);
    border-top-color: var(--text);
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 0 auto 12px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .loader-step {
    font-size: 12px;
    color: var(--text-muted);
  }

  /* RESULTS */
  #results {
    display: none;
  }

  .product-summary {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 14px;
  }

  .product-thumb {
    width: 56px;
    height: 56px;
    border-radius: 6px;
    object-fit: cover;
    border: 1px solid var(--border);
    background: #111;
    flex-shrink: 0;
  }

  .product-meta h2 {
    font-size: 15px;
    font-weight: 600;
    line-height: 1.3;
    margin-bottom: 4px;
  }

  .meta-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .meta-tag {
    border: 1px solid var(--border);
    padding: 2px 6px;
    border-radius: 4px;
  }

  /* TABS */
  .tabs {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    background: var(--surface);
    padding: 4px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 14px;
  }

  .tab-btn {
    background: transparent;
    color: var(--text-muted);
    border: none;
    padding: 8px 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-radius: 4px;
    cursor: pointer;
  }

  .tab-btn.active {
    background: var(--text);
    color: var(--bg);
  }

  .tab-content {
    display: none;
  }

  .tab-content.active {
    display: block;
  }

  /* CARDS IN RESULTS */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    margin-bottom: 12px;
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  .card-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .copy-btn {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
    flex-shrink: 0;
  }

  .copy-btn:active {
    background: var(--text);
    color: var(--bg);
  }

  .hook-box {
    background: #050505;
    border-left: 2px solid var(--text);
    padding: 10px 12px;
    margin-bottom: 12px;
    font-weight: 500;
    font-size: 13px;
  }

  .scene-row {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    margin-bottom: 8px;
    background: #050505;
  }

  .scene-top {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .scene-line {
    margin-bottom: 5px;
    font-size: 13px;
  }

  .scene-line strong {
    color: var(--text-muted);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    margin-right: 4px;
  }

  .prompt-block {
    margin-top: 6px;
    padding: 8px;
    background: #0e0e0e;
    border: 1px dashed #2a2a2a;
    border-radius: 4px;
    font-size: 11px;
    color: #bbbbbb;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }

  .list-item {
    padding: 10px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 8px;
    background: #050505;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }

  .toast {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--text);
    color: var(--bg);
    padding: 8px 16px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: none;
    z-index: 999;
  }
</style>
</head>
<body>

  <header class="header">
    <div class="brand">
      <span class="brand-dot"></span>
      <span>SNIPER // UGC</span>
    </div>
    <span class="badge-status mono">AI ENGINE v2.5</span>
  </header>

  <section class="panel">
    <label class="label mono">01 / ИСТОЧНИК (ССЫЛКА ИЛИ ПРОДУКТ)</label>
    <div class="input-wrap">
      <textarea id="queryInput" class="main-input" placeholder="Вставь ссылку на Wildberries, Ozon, любой сайт, артикул товара или опиши идею словами..."></textarea>
    </div>

    <label class="label mono">02 / ФОРМАТ РОЛИКА</label>
    <div class="chips-scroll" id="formatChips">
      <button class="chip active" data-val="ugc">UGC Обзор</button>
      <button class="chip" data-val="problem_solution">Боль → Решение</button>
      <button class="chip" data-val="unboxing">Распаковка / ASMR</button>
      <button class="chip" data-val="aesthetic">Aesthetic / Визуал</button>
      <button class="chip" data-val="storytelling">Сторителлинг</button>
    </div>

    <label class="label mono">03 / ТОНАЛЬНОСТЬ ПОДАЧИ</label>
    <div class="chips-scroll" id="toneChips">
      <button class="chip active" data-val="native">Нативный (Дружеский)</button>
      <button class="chip" data-val="aggressive">Дерзкий / Триггерный</button>
      <button class="chip" data-val="humor">Юмор / Ирония</button>
      <button class="chip" data-val="luxury">Минимализм / Люкс</button>
    </div>

    <button id="generateBtn" class="btn-primary mono" onclick="runAnalysis()">
      <span>СГЕНЕРИРОВАТЬ СЦЕНАРИИ</span>
      <span>→</span>
    </button>
  </section>

  <div id="loader" class="loader-box">
    <div class="spinner"></div>
    <div id="loaderText" class="loader-step mono">СКАНИРУЮ ИСТОЧНИК И МЕТАДАННЫЕ...</div>
  </div>

  <section id="results">
    <div class="product-summary" id="productHeader"></div>

    <div class="tabs mono">
      <button class="tab-btn active" onclick="switchTab('scripts', this)">Сценарии</button>
      <button class="tab-btn" onclick="switchTab('hooks', this)">Хуки</button>
      <button class="tab-btn" onclick="switchTab('prompts', this)">AI Видео</button>
      <button class="tab-btn" onclick="switchTab('strategy', this)">ЦА & SEO</button>
    </div>

    <div id="tab-scripts" class="tab-content active"></div>
    <div id="tab-hooks" class="tab-content"></div>
    <div id="tab-prompts" class="tab-content"></div>
    <div id="tab-strategy" class="tab-content"></div>
  </section>

  <div id="toast" class="toast mono">СКОПИРОВАНО</div>

<script>
  const tg = window.Telegram?.WebApp;
  if (tg) {
    tg.ready();
    tg.expand();
    if (tg.setHeaderColor) tg.setHeaderColor('#000000');
    if (tg.setBackgroundColor) tg.setBackgroundColor('#000000');
  }

  let selectedFormat = 'ugc';
  let selectedTone = 'native';

  function setupChips(containerId, callback) {
    const container = document.getElementById(containerId);
    container.querySelectorAll('.chip').forEach(btn => {
      btn.addEventListener('click', () => {
        container.querySelectorAll('.chip').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        callback(btn.dataset.val);
      });
    });
  }

  setupChips('formatChips', val => selectedFormat = val);
  setupChips('toneChips', val => selectedTone = val);

  function switchTab(tabName, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + tabName).classList.add('active');
  }

  function copyText(text) {
    navigator.clipboard.writeText(text);
    const toast = document.getElementById('toast');
    toast.style.display = 'block';
    if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
    setTimeout(() => { toast.style.display = 'none'; }, 1400);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  async function runAnalysis() {
    const query = document.getElementById('queryInput').value.trim();
    if (!query) {
      alert('Вставь ссылку, артикул или описание продукта');
      return;
    }

    const btn = document.getElementById('generateBtn');
    const loader = document.getElementById('loader');
    const loaderText = document.getElementById('loaderText');
    const results = document.getElementById('results');

    btn.disabled = true;
    results.style.display = 'none';
    loader.style.display = 'block';

    const steps = [
      'ПАРСИНГ КАРТОЧКИ И МЕТАДАННЫХ...',
      'ВЫЯВЛЕНИЕ БОЛЕЙ И ТРИГГЕРОВ ЦА...',
      'СБОРКА ВИРАЛЬНЫХ ХУКОВ И РАСКАДРОВКИ...',
      'ГЕНЕРАЦИЯ AI VIDEO PROMPTS...'
    ];
    let stepIdx = 0;
    const stepTimer = setInterval(() => {
      stepIdx = (stepIdx + 1) % steps.length;
      loaderText.textContent = steps[stepIdx];
    }, 2200);

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          format_type: selectedFormat,
          tone: selectedTone
        })
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({}));
        throw new Error(errData.detail || 'Ошибка сервера при генерации');
      }

      const data = await resp.json();
      renderResults(data);
      results.style.display = 'block';
      if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
    } catch (err) {
      alert('Ошибка: ' + err.message);
    } finally {
      clearInterval(stepTimer);
      loader.style.display = 'none';
      btn.disabled = false;
    }
  }

  function renderResults(data) {
    // 1. Шапка продукта
    const imgHtml = data.meta_image
      ? `<img class="product-thumb" src="${escapeHtml(data.meta_image)}" onerror="this.style.display='none'">`
      : '';
    const priceTag = data.meta_price ? `<span class="meta-tag mono">${escapeHtml(data.meta_price)}</span>` : '';
    const sourceTag = `<span class="meta-tag mono">${escapeHtml(data.meta_source || 'WEB')}</span>`;
    const catTag = `<span class="meta-tag mono">${escapeHtml(data.category)}</span>`;

    document.getElementById('productHeader').innerHTML = `
      ${imgHtml}
      <div class="product-meta">
        <h2>${escapeHtml(data.product_title)}</h2>
        <div class="meta-tags">${sourceTag}${catTag}${priceTag}</div>
      </div>
    `;

    // 2. Сценарии
    const scriptsHtml = (data.scripts || []).map((sc, idx) => {
      const fullScriptCopy = `СЦЕНАРИЙ: ${sc.title}\nХУК: ${sc.hook}\n\n` +
        sc.scenes.map(s => `[${s.timecode}]\nКадр: ${s.visual}\nОзвучка: ${s.voiceover}\nТекст: ${s.on_screen_text}`).join('\n\n') +
        `\n\nCTA: ${sc.cta}`;

      const scenesList = (sc.scenes || []).map(s => `
        <div class="scene-row">
          <div class="scene-top mono">
            <span>ТАЙМКОД: ${escapeHtml(s.timecode)}</span>
            <span>SCENE</span>
          </div>
          <div class="scene-line"><strong>КАДР:</strong> ${escapeHtml(s.visual)}</div>
          <div class="scene-line"><strong>ГОЛОС:</strong> «${escapeHtml(s.voiceover)}»</div>
          <div class="scene-line"><strong>ТЕКСТ:</strong> ${escapeHtml(s.on_screen_text)}</div>
          <div class="prompt-block mono">
            <span>🎥 AI Prompt: ${escapeHtml(s.ai_video_prompt)}</span>
            <button class="copy-btn" onclick='copyText(${JSON.stringify(s.ai_video_prompt)})'>COPY</button>
          </div>
        </div>
      `).join('');

      return `
        <div class="card">
          <div class="card-header">
            <div>
              <div class="card-title">0${idx + 1} // ${escapeHtml(sc.title)}</div>
              <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">
                Угол: ${escapeHtml(sc.angle)} · Потенциал: <span class="mono" style="color:#fff">${sc.virality_score}%</span>
              </div>
            </div>
            <button class="copy-btn mono" onclick='copyText(${JSON.stringify(fullScriptCopy)})'>КОПИРОВАТЬ ВСЁ</button>
          </div>
          <div class="hook-box">
            <span class="mono" style="font-size:10px; color:var(--text-muted); display:block; margin-bottom:2px;">ХУК (0-2 СЕК):</span>
            «${escapeHtml(sc.hook)}»
          </div>
          ${scenesList}
          <div class="scene-line" style="margin-top:10px; padding-top:8px; border-top:1px solid var(--border);">
            <strong>ПРИЗЫВ (CTA):</strong> ${escapeHtml(sc.cta)}
          </div>
        </div>
      `;
    }).join('');
    document.getElementById('tab-scripts').innerHTML = scriptsHtml;

    // 3. Хуки
    const hooksHtml = (data.viral_hooks || []).map((h, i) => `
      <div class="list-item">
        <div>
          <span class="mono" style="font-size:11px; color:var(--text-muted);">HOOK #0${i + 1}</span>
          <div style="font-weight:500; margin-top:2px;">«${escapeHtml(h)}»</div>
        </div>
        <button class="copy-btn mono" onclick='copyText(${JSON.stringify(h)})'>COPY</button>
      </div>
    `).join('');
    document.getElementById('tab-hooks').innerHTML = `<div class="card"><div class="card-title" style="margin-bottom:12px;">ТРИГГЕРНЫЕ ХУКИ ДЛЯ ПЕРВЫХ 2 СЕКУНД</div>${hooksHtml}</div>`;

    // 4. AI Видео промпты
    const promptsHtml = (data.video_generator_prompts || []).map((p, i) => `
      <div class="list-item">
        <div class="mono" style="font-size:12px;">
          <span style="color:var(--text-muted); display:block; margin-bottom:4px;">KLING / RUNWAY / SORA PROMPT #0${i + 1}</span>
          ${escapeHtml(p)}
        </div>
        <button class="copy-btn mono" onclick='copyText(${JSON.stringify(p)})'>COPY</button>
      </div>
    `).join('');
    document.getElementById('tab-prompts').innerHTML = `<div class="card"><div class="card-title" style="margin-bottom:12px;">ГОТОВЫЕ ПРОМПТЫ ДЛЯ ГЕНЕРАЦИИ B-ROLL</div>${promptsHtml}</div>`;

    // 5. ЦА, Боли, Преимущества и SEO
    const pains = (data.pain_points || []).map(p => `<li style="margin-bottom:6px; margin-left:16px;">${escapeHtml(p)}</li>`).join('');
    const features = (data.killer_features || []).map(f => `<li style="margin-bottom:6px; margin-left:16px;">${escapeHtml(f)}</li>`).join('');
    const hashtags = (data.hashtags || []).join(' ');
    const fullSeo = `${data.seo_caption}\n\n${hashtags}`;

    document.getElementById('tab-strategy').innerHTML = `
      <div class="card">
        <div class="card-header"><div class="card-title">ЦЕЛЕВАЯ АУДИТОРИЯ</div></div>
        <p>${escapeHtml(data.target_audience)}</p>
      </div>
      <div class="card">
        <div class="card-header"><div class="card-title">БОЛИ И ТРИГГЕРЫ ПОКУПАТЕЛЯ</div></div>
        <ul>${pains}</ul>
      </div>
      <div class="card">
        <div class="card-header"><div class="card-title">УБОЙНЫЕ ПРЕИМУЩЕСТВА (USP)</div></div>
        <ul>${features}</ul>
      </div>
      <div class="card">
        <div class="card-header">
          <div class="card-title">SEO ОПИСАНИЕ И ХЭШТЕГИ</div>
          <button class="copy-btn mono" onclick='copyText(${JSON.stringify(fullSeo)})'>COPY SEO</button>
        </div>
        <p style="margin-bottom:10px; white-space:pre-line;">${escapeHtml(data.seo_caption)}</p>
        <div class="mono" style="font-size:12px; color:var(--text-muted);">${escapeHtml(hashtags)}</div>
      </div>
    `;
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    return HTMLResponse(content=FRONTEND_HTML)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
