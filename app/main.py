"""
FastAPI-бекенд для веб-чату з агентом Factum Auto.

Ендпоінти:
- GET  /               -> веб-чат (static/index.html)
- POST /api/chat        -> надіслати повідомлення, отримати відповідь агента
- GET  /api/health       -> перевірка живості сервісу
"""

import asyncio
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from .config import settings
from .agent_graph import run_agent_safe, extract_final_answer

app = FastAPI(title="Factum Auto Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ПРОСТА ІСТОРІЯ ЧАТУ В ПАМ'ЯТІ
#
# УВАГА: це in-memory сховище. Воно зникає при перезапуску контейнера
# і не працює, якщо у вас декілька інстансів/воркерів одночасно.
# Для реального навантаження з кількома користувачами / воркерами
# замініть це на Redis (наприклад, redis.asyncio) — інтерфейс нижче
# спроєктовано так, щоб заміна була локальною (get_history/save_history).
# ============================================================

_sessions: dict[str, list] = defaultdict(list)

# ============================================================
# ПРОСТИЙ RATE LIMIT (по IP, ковзне вікно 60 сек)
# Для продакшену з реальним навантаженням краще використати
# slowapi / redis-based rate limiter.
# ============================================================

_RATE_LIMIT_REQUESTS = 20
_RATE_LIMIT_WINDOW_SECONDS = 60
_request_log: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    log = _request_log[client_ip]

    while log and now - log[0] > _RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()

    if len(log) >= _RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Забагато запитів. Спробуйте, будь ласка, трохи пізніше.",
        )

    log.append(now)


def get_history(session_id: str) -> list:
    return _sessions[session_id]


def save_history(session_id: str, messages: list) -> None:
    # Обрізаємо історію, щоб контекст і вартість запитів до OpenAI не росли необмежено
    _sessions[session_id] = messages[-settings.MAX_HISTORY_MESSAGES:]


# ============================================================
# СХЕМИ ЗАПИТУ / ВІДПОВІДІ
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: str
    stop_reason: str | None = None


# ============================================================
# ЕНДПОІНТИ
# ============================================================

@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    session_id = payload.session_id or str(uuid.uuid4())
    history = get_history(session_id)

    messages = list(history) + [HumanMessage(content=payload.message)]

    # Hard-timeout: якщо soft-механізм всередині run_agent_safe чомусь
    # не спрацював вчасно (наприклад, LLM "завис"), asyncio.wait_for
    # примусово перерве очікування на рівні event loop.
    hard_timeout = settings.AGENT_TIMEOUT_SECONDS + 15

    try:
        trajectory = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent_safe,
                messages,
                settings.AGENT_MAX_STEPS,
                settings.AGENT_TIMEOUT_SECONDS,
            ),
            timeout=hard_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Агент не встиг відповісти вчасно. Спробуйте, будь ласка, ще раз.",
        )

    reply = extract_final_answer(trajectory)

    if trajectory["status"] == "error":
        raise HTTPException(
            status_code=502,
            detail=f"Помилка агента: {trajectory.get('error', 'невідома помилка')}",
        )

    if not reply:
        reply = (
            "Не вдалося сформувати відповідь. Спробуйте, будь ласка, "
            "переформулювати запит."
        )

    # Оновлюємо історію сесії: додаємо повідомлення користувача та відповідь агента
    new_history = messages + [AIMessage(content=reply)]
    save_history(session_id, new_history)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        status=trajectory["status"],
        stop_reason=trajectory.get("stop_reason"),
    )


# ============================================================
# СТАТИЧНИЙ ВЕБ-ЧАТ
# ============================================================

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")
