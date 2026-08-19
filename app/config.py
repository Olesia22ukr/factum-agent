"""
Конфігурація застосунку.

У Colab ключ бралося через google.colab.userdata.get("OPENAI").
У продакшені секрети мають надходити зі змінних середовища —
так працює будь-яка хостинг-платформа (Render, Railway, Fly.io, Docker, тощо).
"""

import os


class Settings:
    OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4.1")

    # Захисні механізми агента
    AGENT_MAX_STEPS: int = int(os.environ.get("AGENT_MAX_STEPS", "20"))
    AGENT_TIMEOUT_SECONDS: float = float(os.environ.get("AGENT_TIMEOUT_SECONDS", "90"))

    # Скрапінг
    HTTP_TIMEOUT_SECONDS: float = float(os.environ.get("HTTP_TIMEOUT_SECONDS", "15"))
    HTTP_MAX_RETRIES: int = int(os.environ.get("HTTP_MAX_RETRIES", "3"))
    SCRAPE_DELAY_SECONDS: float = float(os.environ.get("SCRAPE_DELAY_SECONDS", "0.3"))

    # Ліміт на кількість повідомлень в історії однієї сесії
    # (щоб не роздувати вартість запитів до OpenAI)
    MAX_HISTORY_MESSAGES: int = int(os.environ.get("MAX_HISTORY_MESSAGES", "20"))

    # CORS — список дозволених джерел через кому, "*" за замовчуванням для простоти
    ALLOWED_ORIGINS: list[str] = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

    def validate(self) -> None:
        if not self.OPENAI_API_KEY:
            raise RuntimeError(
                "Змінна середовища OPENAI_API_KEY не встановлена. "
                "Додайте її у налаштуваннях хостингу або у файл .env."
            )


settings = Settings()
