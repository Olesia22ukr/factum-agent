"""
Tools агента: get_car_details, search_cars, assess_car_risk.

Логіка перенесена з оригінального ноутбука без змін по суті.
Додано:
- retry/backoff для HTTP-запитів (сайт може тимчасово відповідати 429/5xx);
- явний User-Agent та timeout (було й раніше, залишено);
- невелика пауза між запитами при пошуку, щоб не навантажувати сайт.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator, ConfigDict
from langchain_core.tools import tool

from .config import settings

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# ============================================================
# ДОПОМІЖНІ ФУНКЦІЇ
# ============================================================

def clean_text(value):
    """Очищає текст від зайвих пробілів."""
    if value is None:
        return None
    return " ".join(str(value).split())


def get_soup(url: str) -> BeautifulSoup:
    """Завантажує HTML-сторінку з retry/backoff та повертає BeautifulSoup."""

    last_error: Exception | None = None

    for attempt in range(1, settings.HTTP_MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=settings.HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")

        except requests.RequestException as e:
            last_error = e
            if attempt < settings.HTTP_MAX_RETRIES:
                time.sleep(0.5 * attempt)  # лінійний backoff
            continue

    raise RuntimeError(
        f"Не вдалося завантажити сторінку {url} "
        f"після {settings.HTTP_MAX_RETRIES} спроб: {last_error}"
    )


def extract_number(text):
    """Витягує перше ціле число з тексту."""
    if not text:
        return None
    match = re.search(r"\d[\d\s,.]*", str(text))
    if not match:
        return None
    number = re.sub(r"[^\d]", "", match.group())
    return int(number) if number else None


def extract_price(value: str | None) -> int | None:
    if value is None:
        return None
    if "не вказано" in value.lower():
        return None
    return extract_number(value)


def extract_mileage(value: str | None) -> int | None:
    if not value:
        return None
    if "не актуальний" in value.lower():
        return None
    return extract_number(value)


def parse_car_page(url: str) -> dict:
    """Завантажує сторінку автомобіля Factum Auto та повертає структуровані дані."""

    soup = get_soup(url)

    lot_info = {}
    specifications = {}

    lot_labels = [
        "VIN", "Лот", "Дата додавання", "Дата аукціону",
        "Тип документа", "Штат документа",
        "Роздрібна вартість", "Поточна ставка",
    ]

    spec_labels = [
        "Пробіг", "Паливо", "Ключі", "Циліндри", "Тип двигуна",
        "Тип кузова", "Привід", "Колір", "Коробка передач",
        "Комплектація", "Стан", "Первинне пошкодження",
        "Вторинне пошкодження", "Класифікація ТЗ",
    ]

    for row in soup.find_all("div"):
        spans = row.find_all("span", recursive=False)
        if len(spans) < 2:
            continue

        label = clean_text(spans[0].get_text(" ", strip=True))
        value = clean_text(spans[1].get_text(" ", strip=True))

        if label in lot_labels and label not in lot_info:
            lot_info[label] = value
        if label in spec_labels and label not in specifications:
            specifications[label] = value

    return {"url": url, "lot_info": lot_info, "specifications": specifications}


# ============================================================
# TOOL №1 — get_car_details
# ============================================================

class CarDetailsInput(BaseModel):
    """Вхідні параметри для отримання інформації про конкретний автомобіль."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(
        ...,
        description="URL сторінки конкретного автомобіля на сайті Factum Auto"
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("URL повинен починатися з http:// або https://")
        if "factum-auto.com" not in value.lower():
            raise ValueError("URL повинен належати сайту factum-auto.com")
        if "/lot/" not in value.lower():
            raise ValueError("URL повинен вести на сторінку конкретного лота")
        return value


@tool(args_schema=CarDetailsInput)
def get_car_details(url: str) -> dict:
    """
    Отримує детальну інформацію про конкретний автомобіль із Factum Auto.

    Використовуй цей інструмент, коли користувач надає URL конкретного
    автомобіля або коли потрібно детальніше проаналізувати знайдений лот.
    """
    soup = get_soup(url)
    page_text = clean_text(soup.get_text(" ", strip=True))

    return {
        "url": url,
        "page_text": page_text[:12000]
    }


# ============================================================
# TOOL №2 — search_cars
# ============================================================

class CarSearchInput(BaseModel):
    """Параметри пошуку автомобілів у каталозі Factum Auto."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    make: str = Field(description="Марка автомобіля, наприклад Toyota, BMW або Ford")
    model: Optional[str] = Field(default=None, description="Модель автомобіля")
    min_year: Optional[int] = Field(default=None, description="Мінімальний рік випуску")
    max_price: Optional[int] = Field(default=None, description="Максимальна поточна ставка в доларах США")
    max_mileage: Optional[int] = Field(default=None, description="Максимальний допустимий пробіг у милях")

    @field_validator("make")
    @classmethod
    def validate_make(cls, value: str) -> str:
        if not value or len(value.strip()) < 2:
            raise ValueError("Не вказано марку автомобіля.")
        return value.strip()

    @field_validator("min_year")
    @classmethod
    def validate_year(cls, value: int | None) -> int | None:
        if value is not None and not 1980 <= value <= 2030:
            raise ValueError("Рік автомобіля повинен бути в межах від 1980 до 2030.")
        return value

    @field_validator("max_price")
    @classmethod
    def validate_price(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Максимальна ставка не може бути від'ємною.")
        return value

    @field_validator("max_mileage")
    @classmethod
    def validate_mileage(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Максимальний пробіг не може бути від'ємним.")
        return value


@tool(args_schema=CarSearchInput)
def search_cars(
    make: str,
    model: str | None = None,
    min_year: int | None = None,
    max_price: int | None = None,
    max_mileage: int | None = None
) -> str:
    """
    Шукає автомобілі в каталозі Factum Auto за критеріями користувача.

    Використовуй цей інструмент, коли користувач хоче знайти автомобілі
    певної марки або моделі та відфільтрувати їх за роком, поточною ставкою
    або пробігом.

    Результат містить total_lots_in_catalog (скільки лотів взагалі є на сайті
    за маркою/моделлю) і lots_scanned (скільки з них реально перевірено на
    відповідність критеріям — сканування обмежене, щоб відповідь не була
    надто довгою). Якщо total_lots_in_catalog > lots_scanned, обов'язково
    повідом користувачу, що переглянуто не всі наявні лоти.

    Поточна ставка на аукціоні не є фінальною ціною автомобіля.
    Значення $0 також не означає, що автомобіль безкоштовний.
    """

    params = {
        "lot-type": "automobile",
        "mark": make.lower(),
        "year-from": min_year if min_year is not None else 1980,
        "year-to": 2026,
    }
    if model:
        params["model"] = model.lower()

    catalog_url = "https://factum-auto.com/catalog?" + urlencode(params)
    soup = get_soup(catalog_url)

    lot_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/lot/" in href:
            lot_links.append(href)
    lot_links = list(dict.fromkeys(lot_links))

    total_lots_found = len(lot_links)

    # Обмежуємо кількість сторінок, які реально скануємо: якщо каталог
    # повертає забагато лотів, повне послідовне сканування може тривати
    # надто довго і хостинг обірве з'єднання. Скануємо перші N.
    lot_links = lot_links[:settings.SEARCH_MAX_LOTS]

    def _process_lot(link: str) -> dict | None:
        """Завантажує й фільтрує один лот. Повертає None, якщо не підходить."""

        full_url = link if link.startswith("http") else "https://factum-auto.com" + link

        try:
            car = parse_car_page(full_url)
            lot_info = car["lot_info"]
            specs = car["specifications"]

            year_match = re.search(r"/lot/\d+-(\d{4})-", full_url)
            year = int(year_match.group(1)) if year_match else None

            price = extract_price(lot_info.get("Поточна ставка"))
            mileage = extract_mileage(specs.get("Пробіг"))

            if min_year is not None and (year is None or year < min_year):
                return None
            if max_price is not None and (price is None or price > max_price):
                return None
            if max_mileage is not None and (mileage is None or mileage > max_mileage):
                return None

            if price == 0:
                price_status = (
                    "Поточна ставка $0. Це не означає, що автомобіль коштує $0. "
                    "Фінальна вартість потребує уточнення."
                )
            elif price is None:
                price_status = "Поточна ставка не вказана. Фінальна вартість потребує уточнення."
            else:
                price_status = f"Поточна ставка: ${price:,}. Це не фінальна вартість автомобіля."

            return {
                "year": year,
                "price": price,
                "price_status": price_status,
                "mileage": mileage,
                "vin": lot_info.get("VIN"),
                "lot": lot_info.get("Лот"),
                "condition": specs.get("Стан"),
                "primary_damage": specs.get("Первинне пошкодження"),
                "secondary_damage": specs.get("Вторинне пошкодження"),
                "keys": specs.get("Ключі"),
                "document_type": lot_info.get("Тип документа"),
                "url": full_url
            }

        except Exception as e:
            # Одна невдала сторінка не повинна валити весь пошук
            print(f"Не вдалося обробити {full_url}: {e}")
            return None

    matched_cars = []

    # Скануємо лоти паралельно замість послідовно — це на порядок швидше
    # і дозволяє вкластись у таймаут хостингу.
    with ThreadPoolExecutor(max_workers=settings.SEARCH_MAX_WORKERS) as executor:
        futures = [executor.submit(_process_lot, link) for link in lot_links]
        for future in as_completed(futures):
            car_data = future.result()
            if car_data is not None:
                matched_cars.append(car_data)

    result = {
        "catalog_url": catalog_url,
        "total_lots_in_catalog": total_lots_found,
        "lots_scanned": len(lot_links),
        "found": len(matched_cars),
        "cars": matched_cars
    }

    return json.dumps(result, ensure_ascii=False, indent=2)


# ============================================================
# TOOL №3 — assess_car_risk
# ============================================================

class CarRiskInput(BaseModel):
    """Параметри для оцінки ризику автомобіля."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    primary_damage: str = Field(description="Первинне пошкодження автомобіля")
    secondary_damage: Optional[str] = Field(default=None, description="Вторинне пошкодження, якщо вказане")
    condition: str = Field(description="Поточний стан автомобіля, наприклад 'Заводиться та їде'")
    keys_available: str = Field(description="Чи є ключі автомобіля в наявності")
    document_type: str = Field(description="Тип документа автомобіля")
    mileage: Optional[int] = Field(default=None, description="Пробіг автомобіля в милях")

    @field_validator("primary_damage", "condition", "keys_available", "document_type")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value or len(value.strip()) < 2:
            raise ValueError("Обов'язковий текстовий параметр не може бути порожнім.")
        return value.strip()

    @field_validator("secondary_damage")
    @classmethod
    def validate_secondary_damage(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value if value else None

    @field_validator("mileage")
    @classmethod
    def validate_mileage(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("Пробіг не може бути від'ємним.")
        return value


@tool(args_schema=CarRiskInput)
def assess_car_risk(
    primary_damage: str,
    condition: str,
    keys_available: str,
    document_type: str,
    secondary_damage: str | None = None,
    mileage: int | None = None
) -> str:
    """
    Оцінює ризик купівлі конкретного автомобіля з аукціону.

    Використовуй цей інструмент після отримання характеристик автомобіля.
    Обов'язково передавай первинне та вторинне пошкодження, якщо вони відомі.
    """

    risk_score = 0
    factors = []

    dangerous_damage = ["пожежа", "затоплення", "біологічне забруднення"]

    primary_lower = primary_damage.lower()
    if any(damage in primary_lower for damage in dangerous_damage):
        risk_score += 3
        factors.append(f"серйозне первинне пошкодження: {primary_damage}")
    elif primary_lower not in ["невідомо", "відсутнє", "немає"]:
        risk_score += 1
        factors.append(f"первинне пошкодження: {primary_damage}")

    if secondary_damage:
        secondary_lower = secondary_damage.lower()
        if any(damage in secondary_lower for damage in dangerous_damage):
            risk_score += 3
            factors.append(f"серйозне вторинне пошкодження: {secondary_damage}")
        elif secondary_lower not in ["невідомо", "відсутнє", "немає"]:
            risk_score += 1
            factors.append(f"вторинне пошкодження: {secondary_damage}")

    if "заводиться та їде" not in condition.lower():
        risk_score += 2
        factors.append(f"стан автомобіля: {condition}")

    if "в наявності" not in keys_available.lower():
        risk_score += 1
        factors.append(f"ключі: {keys_available}")

    risky_documents = ["не підлягає відновленню", "parts only", "certificate of destruction"]
    if any(document in document_type.lower() for document in risky_documents):
        risk_score += 3
        factors.append(f"тип документа потребує додаткової перевірки: {document_type}")

    if mileage is not None and mileage > 150000:
        risk_score += 1
        factors.append(f"високий пробіг: {mileage} миль")

    if risk_score <= 1:
        risk_level = "низький"
    elif risk_score <= 3:
        risk_level = "середній"
    else:
        risk_level = "високий"

    if not factors:
        factors.append("критичних факторів ризику не виявлено")

    return (
        f"Рівень ризику: {risk_level}. "
        f"Risk score: {risk_score}. "
        f"Фактори оцінки: {'; '.join(factors)}."
    )


ALL_TOOLS = [get_car_details, search_cars, assess_car_risk]
