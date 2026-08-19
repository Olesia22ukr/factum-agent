# Factum Auto Agent — production-версія

Веб-чат з AI-агентом для пошуку та оцінки ризику авто з аукціону Factum Auto.
Це production-обгортка над оригінальним ReAct-агентом (LangGraph + LangChain + OpenAI),
розробленим у Colab-ноутбуці `HW1_Factum_ReAct_Agent_final`.

## Що змінено відносно ноутбука

| Проблема в ноутбуці | Рішення тут |
|---|---|
| `google.colab.userdata` для ключа API | Ключ читається зі змінної середовища `OPENAI_API_KEY` (`app/config.py`) |
| `signal.SIGALRM` для timeout — не працює у веб-сервері поза головним потоком | `asyncio.wait_for` + `asyncio.to_thread` у `app/main.py` |
| Немає серверного шару, лише функції в ноутбуці | FastAPI-застосунок з `/api/chat`, сесіями, rate-limit |
| Немає retry для HTTP-запитів до сайту | Backoff-retry у `get_soup()` (`app/tools.py`) |
| Скрипт-стиль з `print`/`assert` по всьому коду | Логіка винесена в модулі `app/tools.py`, `app/agent_graph.py` |
| Немає веб-інтерфейсу | `static/index.html` — простий чат |

## Структура проєкту

```
factum-agent/
├── app/
│   ├── config.py        # налаштування зі змінних середовища
│   ├── tools.py          # 3 tools агента (перенесені з ноутбука)
│   ├── agent_graph.py     # LangGraph, system prompt, safe run
│   └── main.py            # FastAPI: /api/chat, /api/health, роздача фронтенду
├── static/
│   └── index.html          # веб-чат
├── requirements.txt
├── Dockerfile
├── .env.example
└── .dockerignore
```

## Локальний запуск

```bash
cd factum-agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# відкрийте .env і впишіть свій OPENAI_API_KEY

export $(cat .env | xargs)      # Linux/macOS; на Windows задайте змінні вручну
uvicorn app.main:app --reload --port 8000
```

Відкрийте http://localhost:8000 — має завантажитись чат.

Перевірка API напряму:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Знайди Toyota Corolla від 2022 року до 10000 доларів"}'
```

## Деплой у продакшен: Render.com (найпростіший варіант)

Render має безкоштовний план для Docker-застосунків, автодеплой із GitHub і
не потребує керування серверами.

### Крок 1. Покладіть проєкт у GitHub

```bash
cd factum-agent
git init
git add .
git commit -m "Production version of Factum Auto Agent"
git branch -M main
git remote add origin https://github.com/<ваш-логін>/factum-agent.git
git push -u origin main
```

Переконайтесь, що `.env` **не** потрапив у git (він і так у `.dockerignore`,
але додайте `.env` також у `.gitignore`, якщо створюєте репозиторій вручну).

### Крок 2. Створіть Web Service на Render

1. Зареєструйтесь / увійдіть на https://render.com
2. New → Web Service
3. Підключіть ваш GitHub-репозиторій `factum-agent`
4. Render автоматично побачить `Dockerfile` — оберіть **Docker** як Environment
5. Region: оберіть найближчий до ваших користувачів
6. Instance Type: Free (для демо/навчального навантаження достатньо)

### Крок 3. Додайте змінні середовища

У розділі **Environment** сервісу додайте:

| Key | Value |
|---|---|
| `OPENAI_API_KEY` | ваш реальний ключ OpenAI |
| `OPENAI_MODEL` | `gpt-4.1` |
| `AGENT_MAX_STEPS` | `20` |
| `AGENT_TIMEOUT_SECONDS` | `90` |
| `ALLOWED_ORIGINS` | `*` (або ваш домен, якщо є) |

Render сам передає `PORT` — його вручну задавати не треба, `Dockerfile` вже це враховує.

### Крок 4. Deploy

Натисніть **Create Web Service**. Render збере Docker-образ і задеплоїть його.
Через кілька хвилин отримаєте публічний URL на кшталт:

```
https://factum-agent.onrender.com
```

Відкрийте його в браузері — побачите чат.

### Альтернативи Render

- **Railway.app** — так само деплой з GitHub + Dockerfile, теж є безкоштовний ліміт.
- **Fly.io** — трохи більше контролю (регіони, масштабування), деплой командою `fly launch`.
- **VPS + Docker** (будь-який хостинг, напр. Hetzner/DigitalOcean) — якщо потрібен повний контроль:
  ```bash
  docker build -t factum-agent .
  docker run -d -p 80:8000 --env-file .env factum-agent
  ```

## Важливі обмеження поточної версії (чесно про те, що ще не production-grade)

1. **Історія чату — в оперативній памʼяті процесу.** Перезапуск контейнера
   (деплой нової версії, засинання Free-інстансу на Render через неактивність)
   очищує всі сесії. Для реальних користувачів варто винести історію в Redis.
2. **Один інстанс.** Free-план Render/Railway запускає один контейнер —
   цього достатньо для демо чи невеликого потоку користувачів, але
   горизонтальне масштабування (кілька інстансів) вимагатиме зовнішнього
   сховища сесій (той самий Redis), бо зараз сесії живуть у памʼяті конкретного процесу.
3. **Rate-limit — простий, по IP, в памʼяті.** Достатньо, щоб один
   користувач не заспамив агента запитами, але не захищає повноцінно
   від розподілених зловживань. Для публічного продукту розгляньте Cloudflare
   перед сервісом або `slowapi`/Redis-based ліміти.
4. **Скрапінг factum-auto.com** покладається на поточну HTML-структуру сайту
   (пошук `<div><span><span>`). Якщо сайт оновить верстку — `search_cars` і
   `get_car_details` почнуть повертати порожні дані. Перевіряйте це в перших
   тестових запусках у продакшені.
5. **Free-план Render "засинає"** після ~15 хв бездіяльності і перший запит
   після цього буде повільним (холодний старт). Для постійної доступності
   потрібен платний план (там же вирішиться і застереження №1-2, якщо додати Redis-адон).

## Що можна покращити далі

- Redis для сесій та rate-limit (робить застосунок горизонтально масштабованим).
- Стрімінг відповіді агента в чат (Server-Sent Events / WebSocket) замість очікування повної відповіді.
- Логування траєкторій у зовнішній сервіс (напр. LangSmith), а не тільки в памʼяті процесу.
- Автотести (`pytest`) на основі test-кейсів з оригінального ноутбука — зараз вони існують лише як ручні перевірки в Colab.
