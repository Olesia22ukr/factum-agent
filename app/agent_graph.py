"""
Побудова ReAct-агента на LangGraph та безпечний запуск.

Зміни відносно оригінального ноутбука:
1. Ключ OpenAI береться з app.config (env var), а не з google.colab.userdata.
2. Timeout реалізовано через asyncio.wait_for замість signal.SIGALRM —
   у веб-сервері (Uvicorn/FastAPI) запити обробляються НЕ в головному потоці
   головного інтерпретатора, тому signal.alarm там працювати не буде.
   Soft-timeout (перевірка elapsed на кожному кроці) залишено як було,
   а hard-timeout тепер забезпечує asyncio.wait_for на рівні виклику.
3. run_agent_safe тепер приймає готову історію повідомлень (для підтримки
   багатоходового чату), а не лише один запит.
"""

import json
import time
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import settings
from .tools import ALL_TOOLS

# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    model=settings.OPENAI_MODEL,
    temperature=0,
    api_key=settings.OPENAI_API_KEY,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """
Ти AI-агент для пошуку та оцінки автомобілів з аукціону Factum Auto.

ТВОЇ ІНСТРУМЕНТИ:

1. search_cars
   Використовуй для пошуку автомобілів за критеріями користувача.

2. get_car_details
   Використовуй, коли потрібно отримати інформацію про конкретний автомобіль
   за URL його сторінки.

3. assess_car_risk
   Використовуй для оцінки ризику конкретного автомобіля.

ПРАВИЛА:

- Не вигадуй характеристики автомобілів.
- Використовуй дані, отримані через інструменти.
- Якщо користувач просить знайти автомобілі, спочатку використовуй search_cars.
- Під час assess_car_risk передавай і primary_damage, і secondary_damage,
  якщо вони доступні.
- Не ігноруй вторинне пошкодження.
- Не називай поточну ставку фінальною ціною автомобіля.
- Поточна ставка $0 НЕ означає, що автомобіль безкоштовний.
- Не вважай документ "Інше" автоматично безпечним:
  його значення потребує додаткової перевірки.

ФОРМАТ ВІДПОВІДІ ПІСЛЯ ПОШУКУ (search_cars):

Ти НІКОЛИ не показуєш користувачу сирий список знайдених авто без оцінки.
Після виклику search_cars ЗАВЖДИ роби так:

1. Якщо кандидатів більше 4 — обери до 4 найбільш релевантних до запиту
   користувача (за роком, ціною, пробігом), і виклич assess_car_risk
   ОКРЕМО для кожного з обраних. Якщо кандидатів 4 або менше — оціни ризик
   для всіх.
2. Порівняй оцінені варіанти й обери ОДИН найкращий (найнижчий Risk score;
   при рівному score — новіший рік, менший пробіг, нижча ставка).
3. Сформуй відповідь у такій структурі:

   Рекомендований варіант
   — марка/модель, рік, ставка (з поясненням, що це не фінальна ціна),
     пробіг, рівень і Risk score ризику, коротке пояснення чому саме цей;
     обов'язково додай окремий рядок "Посилання: <повний URL>".

   Інші розглянуті варіанти
   — коротко по кожному з решти оцінених автомобілів (рік, ставка, пробіг,
     Risk score), кожен із власним рядком "Посилання: <повний URL>".

   Якщо total_lots_in_catalog > lots_scanned або всього знайдено більше,
   ніж ти оцінив — вкажи це прямо, наприклад: "У каталозі 18 автомобілів
   цієї марки, детально оцінено 4 найбільш релевантних за вашим запитом."

- Тримай опис кожного автомобіля лаконічним: 2-3 короткі речення або
  пункти, без зайвих повторів. Не переказуй сирі дані інструмента дослівно.

- ЗАВЖДИ виводь URL автомобіля повністю (http...) окремим рядком
  "Посилання: <URL>". Ніколи не пиши просто слово "переглянути" чи
  "детальніше" без самого посилання поруч — користувач має змогу натиснути
  саме на URL.
- Це правило форматування не застосовується, якщо користувач просить лише
  список без оцінки, або лише інформацію про один конкретний автомобіль,
  або лише оцінку ризику без пошуку.
- Порівнюй кандидатів насамперед за результатами assess_car_risk,
  а потім враховуй стан, пробіг, рік та поточну ставку.
- Не рекомендуй автомобіль з вищим Risk score,
  якщо доступний кандидат із нижчим Risk score, без чіткого пояснення причини.
- Якщо даних недостатньо, прямо повідом про це.
- Відповідай користувачу українською мовою.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState):
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_agent(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


tool_node = ToolNode(ALL_TOOLS)

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_agent, {"tools": "tools", END: END})
workflow.add_edge("tools", "agent")

graph = workflow.compile()


# ============================================================
# ЛОГУВАННЯ / СЕРІАЛІЗАЦІЯ
# ============================================================

def make_json_safe(value):
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def serialize_message(message: BaseMessage) -> dict:
    result = {
        "type": type(message).__name__,
        "content": make_json_safe(message.content),
    }
    if isinstance(message, AIMessage):
        result["tool_calls"] = [
            {"name": c.get("name"), "args": make_json_safe(c.get("args", {})), "id": c.get("id")}
            for c in message.tool_calls
        ]
    if isinstance(message, ToolMessage):
        result["tool_call_id"] = message.tool_call_id
        result["name"] = getattr(message, "name", None)
    return result


def tool_call_signature(call: dict) -> str:
    return json.dumps(
        {"name": call.get("name"), "args": call.get("args", {})},
        ensure_ascii=False, sort_keys=True, default=str,
    )


def extract_final_answer(trajectory: dict) -> str | None:
    for step in reversed(trajectory.get("steps", [])):
        for message in reversed(step.get("messages", [])):
            if message.get("type") == "AIMessage" and message.get("content"):
                return message["content"]
    return None


# ============================================================
# ЗАХИЩЕНИЙ ЗАПУСК АГЕНТА (синхронний, викликається через asyncio.to_thread)
# ============================================================

def run_agent_safe(
    messages: list[BaseMessage],
    max_steps: int = settings.AGENT_MAX_STEPS,
    timeout_seconds: float = settings.AGENT_TIMEOUT_SECONDS,
) -> dict:
    """
    Запускає LangGraph-агента із soft-механізмами захисту:
    max_steps, elapsed-based timeout, loop detection, логування кроків.

    Hard-timeout (примусове переривання) забезпечується ззовні через
    asyncio.wait_for у main.py — це коректно працює у веб-сервері,
    на відміну від signal.alarm.
    """

    started_at = time.monotonic()

    trajectory = {
        "started_at": datetime.now().isoformat(),
        "max_steps": max_steps,
        "timeout_seconds": timeout_seconds,
        "status": "running",
        "stop_reason": None,
        "steps": [],
    }

    seen_tool_calls = set()
    step_count = 0

    try:
        stream = graph.stream({"messages": messages}, stream_mode="updates")

        for update in stream:
            elapsed = time.monotonic() - started_at

            if elapsed > timeout_seconds:
                trajectory["status"] = "stopped"
                trajectory["stop_reason"] = "timeout"
                break

            step_count += 1
            if step_count > max_steps:
                trajectory["status"] = "stopped"
                trajectory["stop_reason"] = "max_steps"
                break

            for node_name, node_update in update.items():
                step_record = {
                    "step": step_count,
                    "node": node_name,
                    "elapsed_seconds": round(elapsed, 3),
                    "messages": [],
                }

                for message in node_update.get("messages", []):
                    step_record["messages"].append(serialize_message(message))

                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            signature = tool_call_signature(call)
                            if signature in seen_tool_calls:
                                trajectory["status"] = "stopped"
                                trajectory["stop_reason"] = "repeated_tool_call"
                            seen_tool_calls.add(signature)

                trajectory["steps"].append(step_record)
                if trajectory["stop_reason"] is not None:
                    break

            if trajectory["stop_reason"] is not None:
                break

        if trajectory["status"] == "running":
            trajectory["status"] = "completed"
            trajectory["stop_reason"] = "finished"

    except Exception as e:
        trajectory["status"] = "error"
        trajectory["stop_reason"] = type(e).__name__
        trajectory["error"] = str(e)

    trajectory["finished_at"] = datetime.now().isoformat()
    trajectory["duration_seconds"] = round(time.monotonic() - started_at, 3)
    trajectory["step_count"] = step_count

    return trajectory
