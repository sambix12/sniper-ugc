import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import (
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
    LabeledPrice,
    PreCheckoutQuery,
)
from google import genai
from google.genai import types as genai_types

# ================= ВСТАВЬ СВОИ ДАННЫЕ СЮДА =================
BOT_TOKEN = "9221687077:AAGR3LiIIHuGM-sQjg--_FG6bG9eoszmmJs"
GEMINI_API_KEY = "AQ.Ab8RN6Ix2QGeV27cR8MKhh7U3rRZ30mZsK1_H8-uOciL5CJGYw"
APP_URL = "https://eupqz-109-252-187-34.run.pinggy-free.link"  # например: https://xxxx.ngrok-free.app
PRICE_STARS = 25  # Стоимость сценария в Telegram Stars
# ==========================================================

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
ai_client = genai.Client(api_key=GEMINI_API_KEY)

generations_db = {}

SYSTEM_PROMPT = """
Ты — циничный арбитражник и продюсер вирального UGC-контента для TikTok / Reels.
Тебе дают товар или оффер. Преврати его в жесткую связку для удержания зрителя.
Ответь ТОЛЬКО валидным JSON со следующей структурой:
{
  "title": "Хлесткое название креатива",
  "audience_pain": "Скрытая боль аудитории",
  "free_hooks": [
    "Хук 1 (Хейт-байт)",
    "Хук 2 (Визуальный шок)",
    "Хук 3 (Разнос аналогов)"
  ],
  "full_script": "Покадровая раскадровка на 60 сек с таймкодами",
  "kling_prompt": "Ultra-realistic 4k prompt in English for Kling/Runway AI video generation",
  "elevenlabs_prompt": "Текст озвучки с паузами [sigh] и живым дыханием",
  "comment_bait": "Какую ошибку намеренно сделать в кадре/тексте, чтобы набежали комментаторы крутить видео"
}
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(dp.start_polling(bot))
    yield
    await bot.session.close()

app = FastAPI(lifespan=lifespan)

class GenRequest(BaseModel):
    product_text: str
    user_id: int

@app.post("/api/generate")
async def generate(req: GenRequest):
    if not req.product_text.strip():
        raise HTTPException(status_code=400, detail="Пустой ввод")
    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"Товар/Оффер:\n{req.product_text}",
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.85,
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        gen_id = f"{req.user_id}_{len(generations_db) + 1}"
        generations_db[gen_id] = {"data": data, "paid": False, "user_id": req.user_id}
        return {
            "gen_id": gen_id,
            "title": data.get("title"),
            "pain": data.get("audience_pain"),
            "hooks": data.get("free_hooks", []),
            "paid": False
        }
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Ошибка ИИ")

@app.post("/api/buy")
async def buy(gen_id: str):
    if gen_id not in generations_db:
        raise HTTPException(status_code=404, detail="Не найдено")
    
    invoice_link = await bot.create_invoice_link(
        title="🎬 Виральный UGC Пак",
        description="Полный сценарий + Промпты для Kling и ElevenLabs",
        payload=f"pay_{gen_id}",
        currency="XTR",
        prices=[LabeledPrice(label="Сценарий + Промпты", amount=PRICE_STARS)],
        provider_token=""
    )
    return {"invoice_link": invoice_link}

@app.get("/api/result/{gen_id}")
async def get_result(gen_id: str):
    item = generations_db.get(gen_id)
    if not item or not item["paid"]:
        raise HTTPException(status_code=403, detail="Не оплачено")
    return item["data"]

@dp.message(CommandStart())
async def start(message: types.Message):
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🚀 Запустить Mini App", web_app=WebAppInfo(url=APP_URL))]
    ])
    await message.answer(
        "🔥 **Салют! Это твой генератор виральных связок под TikTok/Reels.**\n\n"
        "Жми кнопку ниже, чтобы открыть приложение:",
        reply_markup=kb,
        parse_mode="Markdown"
    )

@dp.pre_checkout_query()
async def pre_check(q: PreCheckoutQuery):
    await bot.answer_pre_checkout_query(q.id, ok=True)

@dp.message(F.successful_payment)
async def success_pay(msg: types.Message):
    payload = msg.successful_payment.invoice_payload
    gen_id = payload.replace("pay_", "")
    if gen_id in generations_db:
        generations_db[gen_id]["paid"] = True
    await msg.answer("⭐️ Оплата принята! Вернись в Mini App — всё разблокировано.")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
      <title>SniperUGC</title>
      <script src="https://telegram.org/js/telegram-web-app.js"></script>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-zinc-950 text-white min-h-screen p-4 font-sans">
      <div class="max-w-md mx-auto space-y-4">
        <header class="text-center pt-2">
          <h1 class="text-2xl font-black tracking-tight text-amber-400">⚡️ SNIPER UGC</h1>
          <p class="text-xs text-zinc-400">AI-генератор виральных TikTok & Reels креативов</p>
        </header>

        <div class="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 space-y-3">
          <label class="text-xs font-semibold uppercase text-zinc-400">Товар или оффер:</label>
          <textarea id="productText" rows="3" placeholder="Вставь ссылку на WB/Ozon/Ali или опиши товар..." class="w-full bg-zinc-950 border border-zinc-700 rounded-xl p-3 text-sm focus:border-amber-400 focus:outline-none"></textarea>
          <button id="genBtn" onclick="generate()" class="w-full bg-amber-400 hover:bg-amber-300 text-black font-bold py-3 rounded-xl transition">Сгенерировать хуки 🔥</button>
        </div>

        <div id="output" class="hidden bg-zinc-900 border border-zinc-800 rounded-2xl p-4 space-y-3">
          <h2 id="outTitle" class="font-bold text-amber-400 text-base"></h2>
          <p id="outPain" class="text-xs text-zinc-400 italic"></p>
          <div class="space-y-1">
            <p class="text-xs font-semibold text-zinc-300">🎯 Виральные хуки (3 сек):</p>
            <ul id="hooksList" class="text-xs text-zinc-200 list-disc list-inside space-y-1"></ul>
          </div>
          
          <div id="lockedBlock" class="pt-3 border-t border-zinc-800 space-y-2 text-center">
            <p class="text-xs text-zinc-400">🔒 Полный сценарий, раскадровка и промпты для Kling / ElevenLabs закрыты.</p>
            <button onclick="buy()" class="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2.5 rounded-xl transition">⭐️ Открыть всё за Stars</button>
          </div>

          <div id="unlockedBlock" class="hidden pt-3 border-t border-zinc-800 space-y-3">
            <div>
              <p class="text-xs font-semibold text-amber-400">🎬 Полный сценарий (60 сек):</p>
              <pre id="scriptText" class="text-xs text-zinc-300 whitespace-pre-wrap bg-zinc-950 p-2 rounded-lg mt-1"></pre>
            </div>
            <div>
              <p class="text-xs font-semibold text-indigo-400">🎥 Промпт для Kling / Runway:</p>
              <pre id="klingText" class="text-xs text-zinc-300 whitespace-pre-wrap bg-zinc-950 p-2 rounded-lg mt-1 select-all"></pre>
            </div>
            <div>
              <p class="text-xs font-semibold text-emerald-400">💬 Байт для срача в комментах:</p>
              <p id="baitText" class="text-xs text-zinc-300 bg-zinc-950 p-2 rounded-lg mt-1"></p>
            </div>
          </div>
        </div>
      </div>

      <script>
        const tg = window.Telegram?.WebApp;
        if (tg) { tg.ready(); tg.expand(); }
        let currentGenId = null;

        async function generate() {
          const text = document.getElementById('productText').value;
          if (!text) return alert('Напиши товар!');
          const btn = document.getElementById('genBtn');
          btn.disabled = true;
          btn.innerText = '⚡️ Генерируем связку...';

          try {
            const userId = tg?.initDataUnsafe?.user?.id || 12345;
            const res = await fetch('/api/generate', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({product_text: text, user_id: userId})
            });
            const data = await res.json();
            currentGenId = data.gen_id;

            document.getElementById('outTitle').innerText = data.title;
            document.getElementById('outPain').innerText = 'Боль: ' + data.pain;
            const hl = document.getElementById('hooksList');
            hl.innerHTML = '';
            data.hooks.forEach(h => {
              const li = document.createElement('li');
              li.innerText = h;
              hl.appendChild(li);
            });

            document.getElementById('output').classList.remove('hidden');
          } catch (e) {
            alert('Ошибка при генерации!');
          } finally {
            btn.disabled = false;
            btn.innerText = 'Сгенерировать хуки 🔥';
          }
        }

        async function buy() {
          if (!currentGenId) return;
          try {
            const res = await fetch('/api/buy?gen_id=' + currentGenId, {method: 'POST'});
            const data = await res.json();
            if (tg && tg.openInvoice) {
              tg.openInvoice(data.invoice_link, async (status) => {
                if (status === 'paid') {
                  loadFull();
                }
              });
            } else {
              window.open(data.invoice_link, '_blank');
            }
          } catch(e) {
            alert('Ошибка открытия счета');
          }
        }

        async function loadFull() {
          const res = await fetch('/api/result/' + currentGenId);
          const data = await res.json();
          document.getElementById('scriptText').innerText = data.full_script;
          document.getElementById('klingText').innerText = data.kling_prompt;
          document.getElementById('baitText').innerText = data.comment_bait;
          document.getElementById('lockedBlock').classList.add('hidden');
          document.getElementById('unlockedBlock').classList.remove('hidden');
        }
      </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
