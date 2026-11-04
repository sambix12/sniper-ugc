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
# НАСТРОЙКИ И ИНИЦИАЛИЗАЦИЯ
# =====================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
APP_URL = os.getenv("RENDER_EXTERNAL_URL", "https://sniper-ugc.onrender.com").strip()

bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None
dp = Dispatcher()
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class AnalyzeRequest(BaseModel):
    query: str
    format_type: str = "ugc"
    tone: str = "native"

# =====================================================================
# БЕЗОПАСНЫЙ СБОРЩИК ДАННЫХ (WILDBERRIES / ССЫЛКИ / ТЕКСТ)
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


async def parse_wildberries_safe(nm_id: int, client: httpx.AsyncClient) -> dict:
    vol = nm_id // 100000
    part = nm_id // 1000
    host = get_wb_basket_host(vol)
    base_url = f"{host}/vol{vol}/part{part}/{nm_id}"

    info = {
        "source": "Wildberries",
        "id": str(nm_id),
        "title": f"Товар WB #{nm_id}",
        "brand": "",
        "description": "",
        "features": [],
        "price": "",
        "image": f"{base_url}/images/big/1.webp"
    }

    try:
        r_card = await client.get(f"{base_url}/info/ru/card.json", timeout=6.0)
        if r_card.status_code == 200:
            data = r_card.json()
            info["title"] = data.get("imt_name") or info["title"]
            info["brand"] = data.get("selling", {}).get("brand_name", "")
            info["description"] = data.get("description", "")
            opts = data.get("options", [])
            info["features"] = [f"{o.get('name')}: {o.get('value')}" for o in opts[:10]]
    except Exception as e:
        logging.warning(f"WB card.json warning: {e}")

    try:
        card_api = f"https://card.wb.ru/cards/v2/detail?appType=1&curr=rub&dest=-1257786&spp=30&nm={nm_id}"
        r_detail = await client.get(card_api, timeout=6.0)
        if r_detail.status_code == 200:
            prods = r_detail.json().get("data", {}).get("products", [])
            if prods:
                p = prods[0]
                if not info["brand"]:
                    info["brand"] = p.get("brand", "")
                if "Товар WB" in info["title"]:
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

    # 1. Проверка Wildberries (по ссылке или по числовому артикулу)
    wb_match = re.search(r"wildberries\.ru/catalog/(\d+)", raw)
    num_match = re.fullmatch(r"\d{6,12}", raw)
    if wb_match or num_match:
        nm_id = int(wb_match.group(1) if wb_match else raw)
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, verify=False) as client:
            return await parse_wildberries_safe(nm_id, client)

    # 2. Проверка веб-ссылки (Ozon, Яндекс Маркет, лендинги, интернет-магазины)
    url_match = re.search(r"(https?://[^\s]+)", raw)
    if url_match:
        url = url_match.group(1)
        fallback_data = {
            "source": "Web Link",
            "url": url,
            "title": f"Продукт: {url}",
            "description": "",
            "features": [],
            "image": "",
            "raw_text": f"Ссылка на товар: {url}\nПользовательский контекст: {raw}"
        }
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=8.0, verify=False) as client:
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
                    fallback_data["raw_text"] += f"\nКонтент сайта: {clean[:2500]}"
        except Exception as e:
            logging.warning(f"URL parsing skipped due to: {e}")
        return fallback_data

    # 3. Прямой текстовый ввод
    return {
        "source": "Текст",
        "title": raw[:80],
        "description": raw,
        "features": [],
        "image": ""
    }


# =====================================================================
# ГЕНЕРАЦИЯ UGC-ВОРОНКИ ЧЕРЕЗ GEMINI
# =====================================================================
async def generate_ugc_payload(data: dict, format_type: str, tone: str) -> dict:
    if not ai_client:
        raise ValueError("GEMINI_API_KEY не задан в переменных окружения на Render. Добавь его во вкладке Environment.")

    format_labels = {
        "ugc": "Нативный честный отзыв реального пользователя",
        "problem_solution": "Острая боль ЦА -> Спасение с помощью товара",
        "unboxing": "Быстрая распаковка / ASMR / Эффект первого впечатления",
        "aesthetic": "Эстетика / TikTok Lifestyle / Минималистичный визуал",
        "storytelling": "Жизненная история / Провокация / Неожиданный финал"
    }
    tone_labels = {
        "native": "Разговорный, нативный, доверительный",
        "aggressive": "Дерзкий, кликбейтный, высокий темп, жесткие триггеры",
        "humor": "Ирония, легкий стеб, житейский юмор",
        "luxury": "Дорогой, спокойный, акцент на статусе и качестве"
    }

    system_instruction = """
Ты — топ-продюсер вирального контента для TikTok, Reels и Shorts.
Твоя задача — проанализировать товар/ссылку и сгенерировать готовую виральную воронку продвижения.
Ты ВСЕГДА возвращаешь результат СТРОГО в виде валидного JSON без markdown-разметки (без ```json).

ФОРМАТ JSON:
{
  "product_title": "Название товара",
  "category": "Категория",
  "target_audience": "Кто покупает",
  "pain_points": ["Боль 1", "Боль 2", "Боль 3"],
  "killer_features": ["Фишка 1", "Фишка 2", "Фишка 3"],
  "viral_hooks": [
    "Хук 1 (первые 2 сек)",
    "Хук 2",
    "Хук 3",
    "Хук 4",
    "Хук 5"
  ],
  "scripts": [
    {
      "title": "Название сценария",
      "angle": "Угол захода (например: Через разоблачение)",
      "virality_score": 96,
      "hook": "Текст хука",
      "scenes": [
        {
          "timecode": "0:00-0:03",
          "visual": "Что в кадре",
          "voiceover": "Что говорится голосом",
          "on_screen_text": "Текст на экране плашкой",
          "ai_video_prompt": "Cinematic b-roll prompt in English for Kling or Runway"
        }
      ],
      "cta": "Призыв к действию"
    }
  ],
  "video_generator_prompts": [
    "Prompt 1 for Kling/Runway (English)",
    "Prompt 2 for Kling/Runway (English)",
    "Prompt 3 for Kling/Runway (English)"
  ],
  "seo_caption": "Продающий текст под ролик",
  "hashtags": ["#fyp", "#вайлдберриз"]
}
"""

    user_prompt = f"""
ДАННЫЕ О ПРОДУКТЕ:
{json.dumps(data, ensure_ascii=False, indent=2)}

ФОРМАТ ВИДЕО: {format_labels.get(format_type, format_type)}
ТОНАЛЬНОСТЬ: {tone_labels.get(tone, tone)}

Сделай 3 детализированных сценария с раскадровкой, 5 хуков, 3 качественных промпта для ИИ-видео (Kling/Runway), выдели боли ЦА и SEO-описание.
"""

    resp = await asyncio.to_thread(
        ai_client.models.generate_content,
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7,
            response_mime_type="application/json"
        )
    )

    clean_json = resp.text.strip()
    if clean_json.startswith("```"):
        clean_json = re.sub(r"^```json\s*", "", clean_json)
        clean_json = re.sub(r"^```\s*", "", clean_json)
        clean_json = re.sub(r"\s*```$", "", clean_json)

    parsed_result = json.loads(clean_json)
    parsed_result["meta_source"] = data.get("source", "Web")
    parsed_result["meta_image"] = data.get("image", "")
    parsed_result["meta_price"] = data.get("price", "")
    parsed_result["meta_brand"] = data.get("brand", "")
    return parsed_result


# =====================================================================
# TELEGRAM BOT LOGIC
# =====================================================================
@dp.message(Command("start"))
async def bot_start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="⚡ Открыть SNIPER UGC", web_app=WebAppInfo(url=APP_URL))]
    ])
    await message.answer(
        "<b>SNIPER // UGC GROWTH ENGINE</b>\n\n"
        "Отправь артикул или ссылку на Wildberries / любой сайт, чтобы получить готовые сценарии и раскадровки, "
        "или открой студию приложения по кнопке ниже:",
        reply_markup=kb,
        parse_mode="HTML"
    )

@dp.message()
async def bot_quick_message(message: types.Message):
    txt = (message.text or "").strip()
    if not txt:
        return
    wait_msg = await message.answer("🔍 <b>Анализирую продукт и собираю воронку...</b>", parse_mode="HTML")
    try:
        data = await extract_product_data(txt)
        res = await generate_ugc_payload(data, "ugc", "native")
        s1 = res["scripts"][0]
        hooks = "\n".join([f"• <i>«{h}»</i>" for h in res["viral_hooks"][:3]])
        
        reply = (
            f"🎯 <b>{res['product_title']}</b>\n\n"
            f"🔥 <b>ТОП-ХУКИ:</b>\n{hooks}\n\n"
            f"🎬 <b>СЦЕНАРИЙ: {s1['title']} ({s1.get('virality_score', 95)}% виральность)</b>\n"
            f"<b>Хук:</b> {s1['hook']}\n\n"
            f"📢 <b>CTA:</b> {s1['cta']}"
        )
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⚡ Открыть полную раскадровку", web_app=WebAppInfo(url=APP_URL))]
        ])
        await wait_msg.edit_text(reply, reply_markup=kb, parse_mode="HTML")
    except Exception as e:
        logging.error(f"Telegram parse err: {e}")
        await wait_msg.edit_text(f"⚠️ Ошибка: {e}")


# =====================================================================
# FASTAPI APP & HTML INTERFACE
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    if bot:
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            task = asyncio.create_task(dp.start_polling(bot, handle_signals=False))
            logging.info("Bot Polling started.")
        except Exception as e:
            logging.error(f"Polling start failed: {e}")
    yield
    if task:
        task.cancel()
    if bot:
        await bot.session.close()

app = FastAPI(title="SNIPER UGC", lifespan=lifespan)

@app.head("/")
async def head_ping():
    return HTMLResponse(content="", status_code=200)

@app.post("/api/analyze")
async def api_analyze_endpoint(req: AnalyzeRequest):
    if not req.query or len(req.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Введите ссылку, артикул или описание товара")
    try:
        data = await extract_product_data(req.query)
        result = await generate_ugc_payload(data, req.format_type, req.tone)
        return result
    except Exception as e:
        logging.error(f"API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

FRONTEND_UI = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>SNIPER // UGC</title>
<script src="[https://telegram.org/js/telegram-web-app.js](https://telegram.org/js/telegram-web-app.js)"></script>
<link href="[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap)" rel="stylesheet">
<style>
  :root {
    --bg: #000000;
    --surface: #0a0a0a;
    --border: #222222;
    --text: #ffffff;
    --text-muted: #888888;
    --radius: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    padding: 16px 16px 80px;
    min-height: 100vh;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 14px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
  }
  .brand { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 15px; }
  .dot { width: 8px; height: 8px; background: #fff; border-radius: 50%; box-shadow: 0 0 8px #fff; }
  .tag { font-size: 10px; border: 1px solid var(--border); padding: 3px 6px; border-radius: 4px; color: var(--text-muted); }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    margin-bottom: 16px;
  }
  .lbl { font-size: 11px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; display: block; }
  textarea.inp {
    width: 100%;
    min-height: 72px;
    background: #030303;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: #fff;
    padding: 10px;
    font-size: 13px;
    font-family: inherit;
    resize: none;
    outline: none;
    margin-bottom: 12px;
  }
  textarea.inp:focus { border-color: #fff; }
  .chips { display: flex; gap: 6px; overflow-x: auto; margin-bottom: 12px; scrollbar-width: none; }
  .chips::-webkit-scrollbar { display: none; }
  .chip {
    flex-shrink: 0;
    background: #030303;
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
  }
  .chip.active { background: #fff; color: #000; border-color: #fff; font-weight: 600; }
  .btn-submit {
    width: 100%;
    background: #fff;
    color: #000;
    border: none;
    padding: 12px;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 8px;
  }
  .btn-submit:disabled { opacity: 0.4; cursor: not-allowed; }
  .loader {
    display: none;
    padding: 24px;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    text-align: center;
    margin-bottom: 16px;
  }
  .spinner {
    width: 22px;
    height: 22px;
    border: 2px solid var(--border);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin: 0 auto 10px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  #results { display: none; }
  .prod-head {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 14px;
  }
  .prod-thumb { width: 50px; height: 50px; border-radius: 6px; object-fit: cover; background: #111; flex-shrink: 0; }
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
    padding: 8px 2px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    border-radius: 4px;
    cursor: pointer;
  }
  .tab-btn.active { background: #fff; color: #000; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
    margin-bottom: 12px;
  }
  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }
  .copy-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 10px;
    cursor: pointer;
  }
  .copy-btn:active { background: #fff; color: #000; }
  .hook-box {
    background: #000;
    border-left: 2px solid #fff;
    padding: 8px 10px;
    font-size: 13px;
    margin-bottom: 10px;
  }
  .scene-item {
    background: #000;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .prompt-box {
    margin-top: 6px;
    padding: 6px 8px;
    background: #0a0a0a;
    border: 1px dashed #222;
    border-radius: 4px;
    font-size: 11px;
    color: #aaa;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
  }
  .toast {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: #fff;
    color: #000;
    padding: 6px 14px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    display: none;
    z-index: 1000;
  }
</style>
</head>
<body>
  <div class="header">
    <div class="brand"><div class="dot"></div><span>SNIPER // UGC</span></div>
    <span class="tag mono">V2.0 STABLE</span>
  </div>

  <div class="panel">
    <span class="lbl mono">Входной источник (Ссылка / Артикул / Описание)</span>
    <textarea id="queryInp" class="inp" placeholder="Вставь ссылку на Wildberries, Ozon, любой сайт или артикул..."></textarea>

    <span class="lbl mono">Формат ролика</span>
    <div class="chips" id="fmtChips">
      <button class="chip active" data-v="ugc">UGC Обзор</button>
      <button class="chip" data-v="problem_solution">Боль → Решение</button>
      <button class="chip" data-v="unboxing">Распаковка / ASMR</button>
      <button class="chip" data-v="aesthetic">Aesthetic</button>
      <button class="chip" data-v="storytelling">Сторителлинг</button>
    </div>

    <span class="lbl mono">Тональность</span>
    <div class="chips" id="toneChips">
      <button class="chip active" data-v="native">Нативный</button>
      <button class="chip" data-v="aggressive">Дерзкий / Триггер</button>
      <button class="chip" data-v="humor">Юмор / Мем</button>
      <button class="chip" data-v="luxury">Премиум / Люкс</button>
    </div>

    <button id="submitBtn" class="btn-submit mono" onclick="startGeneration()">
      <span>СГЕНЕРИРОВАТЬ СЦЕНАРИИ</span> <span>→</span>
    </button>
  </div>

  <div id="loader" class="loader">
    <div class="spinner"></div>
    <div id="loaderTxt" class="mono" style="font-size:11px; color:var(--text-muted);">ИЗВЛЕЧЕНИЕ ДАННЫХ И СЦЕНАРИЕВ...</div>
  </div>

  <div id="results">
    <div class="prod-head" id="pHeader"></div>
    <div class="tabs mono">
      <button class="tab-btn active" onclick="setTab('scripts', this)">Сценарии</button>
      <button class="tab-btn" onclick="setTab('hooks', this)">Хуки</button>
      <button class="tab-btn" onclick="setTab('prompts', this)">AI Видео</button>
      <button class="tab-btn" onclick="setTab('seo', this)">ЦА & SEO</button>
    </div>
    <div id="tab-scripts" class="tab-content active"></div>
    <div id="tab-hooks" class="tab-content"></div>
    <div id="tab-prompts" class="tab-content"></div>
    <div id="tab-seo" class="tab-content"></div>
  </div>

  <div id="toast" class="toast mono">СКОПИРОВАНО</div>

<script>
  const tg = window.Telegram?.WebApp;
  if (tg) {
    tg.ready();
    tg.expand();
    if (tg.setHeaderColor) tg.setHeaderColor('#000000');
    if (tg.setBackgroundColor) tg.setBackgroundColor('#000000');
  }

  let curFmt = 'ugc';
  let curTone = 'native';

  function bindChips(id, cb) {
    document.querySelectorAll('#' + id + ' .chip').forEach(c => {
      c.onclick = () => {
        document.querySelectorAll('#' + id + ' .chip').forEach(x => x.classList.remove('active'));
        c.classList.add('active');
        cb(c.dataset.v);
      };
    });
  }
  bindChips('fmtChips', v => curFmt = v);
  bindChips('toneChips', v => curTone = v);

  function setTab(name, btn) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
  }

  function copy(txt) {
    navigator.clipboard.writeText(txt);
    const t = document.getElementById('toast');
    t.style.display = 'block';
    if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
    setTimeout(() => t.style.display = 'none', 1200);
  }

  function esc(s) {
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  async function startGeneration() {
    const q = document.getElementById('queryInp').value.trim();
    if (!q) {
      alert('Укажи ссылку или артикул');
      return;
    }
    const btn = document.getElementById('submitBtn');
    const ldr = document.getElementById('loader');
    const res = document.getElementById('results');

    btn.disabled = true;
    res.style.display = 'none';
    ldr.style.display = 'block';

    try {
      const resp = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, format_type: curFmt, tone: curTone })
      });

      const payload = await resp.json();
      if (!resp.ok) {
        throw new Error(payload.detail || 'Не удалось обработать запрос');
      }
      render(payload);
      res.style.display = 'block';
      if (tg?.HapticFeedback) tg.HapticFeedback.notificationOccurred('success');
    } catch (e) {
      alert('Ошибка: ' + e.message);
    } finally {
      ldr.style.display = 'none';
      btn.disabled = false;
    }
  }

  function render(d) {
    const imgHtml = d.meta_image ? `<img class="prod-thumb" src="${esc(d.meta_image)}" onerror="this.style.display='none'">` : '';
    const price = d.meta_price ? `<span class="tag mono">${esc(d.meta_price)}</span>` : '';
    document.getElementById('pHeader').innerHTML = `
      ${imgHtml}
      <div>
        <div style="font-weight:600; font-size:14px; margin-bottom:4px;">${esc(d.product_title)}</div>
        <div style="display:flex; gap:6px;">
          <span class="tag mono">${esc(d.meta_source || 'WEB')}</span>
          <span class="tag mono">${esc(d.category || 'Товар')}</span>
          ${price}
        </div>
      </div>
    `;

    // 1. Сценарии
    document.getElementById('tab-scripts').innerHTML = (d.scripts || []).map((sc, i) => {
      const full = `Сценарий: ${sc.title}\nХук: ${sc.hook}\n` + (sc.scenes || []).map(s => `[${s.timecode}] ${s.visual} | Голос: ${s.voiceover}`).join('\n') + `\nCTA: ${sc.cta}`;
      const scenes = (sc.scenes || []).map(s => `
        <div class="scene-item">
          <div class="mono" style="color:var(--text-muted); font-size:10px; margin-bottom:4px;">${esc(s.timecode)}</div>
          <div><strong>Кадр:</strong> ${esc(s.visual)}</div>
          <div style="margin:3px 0;"><strong>Голос:</strong> «${esc(s.voiceover)}»</div>
          <div><strong>Текст:</strong> ${esc(s.on_screen_text)}</div>
          <div class="prompt-box mono">
            <span>AI: ${esc(s.ai_video_prompt)}</span>
            <button class="copy-btn" onclick='copy(${JSON.stringify(s.ai_video_prompt)})'>COPY</button>
          </div>
        </div>
      `).join('');

      return `
        <div class="card">
          <div class="card-top">
            <div>
              <div style="font-weight:700; text-transform:uppercase;">0${i+1} // ${esc(sc.title)}</div>
              <div class="mono" style="font-size:10px; color:var(--text-muted);">${esc(sc.angle)} · ${sc.virality_score}% VIRALITY</div>
            </div>
            <button class="copy-btn mono" onclick='copy(${JSON.stringify(full)})'>ВЕСЬ СЦЕНАРИЙ</button>
          </div>
          <div class="hook-box"><strong>ХУК:</strong> «${esc(sc.hook)}»</div>
          ${scenes}
          <div style="font-size:12px; margin-top:8px;"><strong>CTA:</strong> ${esc(sc.cta)}</div>
        </div>
      `;
    }).join('');

    // 2. Хуки
    document.getElementById('tab-hooks').innerHTML = `
      <div class="card">
        ${(d.viral_hooks || []).map((h, i) => `
          <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid var(--border);">
            <div style="font-size:13px;"><strong>#0${i+1}</strong> «${esc(h)}»</div>
            <button class="copy-btn mono" onclick='copy(${JSON.stringify(h)})'>COPY</button>
          </div>
        `).join('')}
      </div>
    `;

    // 3. AI Видео
    document.getElementById('tab-prompts').innerHTML = `
      <div class="card">
        ${(d.video_generator_prompts || []).map((p, i) => `
          <div class="prompt-box mono" style="margin-bottom:8px;">
            <span>${esc(p)}</span>
            <button class="copy-btn" onclick='copy(${JSON.stringify(p)})'>COPY</button>
          </div>
        `).join('')}
      </div>
    `;

    // 4. ЦА & SEO
    const ht = (d.hashtags || []).join(' ');
    const seoFull = `${d.seo_caption}\n\n${ht}`;
    document.getElementById('tab-seo').innerHTML = `
      <div class="card">
        <div class="card-top"><strong>ЦЕЛЕВАЯ АУДИТОРИЯ</strong></div>
        <div style="font-size:13px;">${esc(d.target_audience)}</div>
      </div>
      <div class="card">
        <div class="card-top">
          <strong>SEO ОПИСАНИЕ И ТЕГИ</strong>
          <button class="copy-btn mono" onclick='copy(${JSON.stringify(seoFull)})'>COPY SEO</button>
        </div>
        <div style="font-size:13px; margin-bottom:8px; white-space:pre-line;">${esc(d.seo_caption)}</div>
        <div class="mono" style="font-size:11px; color:var(--text-muted);">${esc(ht)}</div>
      </div>
    `;
  }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    return HTMLResponse(content=FRONTEND_UI)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
