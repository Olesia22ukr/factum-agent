FROM python:3.12-slim

WORKDIR /srv

# Системні залежності для requests/bs4 не потрібні поза стандартними,
# але лишаємо шар для кешування pip-пакетів окремо від коду
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static

# Render / Railway / Fly.io передають порт через змінну PORT
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
