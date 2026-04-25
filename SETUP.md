# Инструкция по запуску проекта

## Требования
- Python 3.10+
- pip или conda

## Установка

### 1. Клонирование репозитория
```bash
git clone <your-repo-url>
cd project_twitch
```

### 2. Создание виртуального окружения
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
```

## Загрузка данных

### Вариант 1: Вручную
1. Перейдите на [Kaggle](https://www.kaggle.com/datasets/aayushmishra1512/twitchdata)
2. Скачайте датасет
3. Распакуйте CSV файл в `data/raw/twitchdata.csv`

### Вариант 2: Через Kaggle API
```bash
# Настройте Kaggle API (kaggle.json в ~/.kaggle/)
kaggle datasets download -d aayushmishra1512/twitchdata -p data/raw --unzip
```

## Запуск проекта

### 1. Exploratory Data Analysis (EDA)
```bash
python run_eda.py
```

### 2. Визуализация данных
```bash
python visualize_data.py
```

### 3. Обучение моделей
```bash
python train_models.py
```

## Проверка кода с помощью линтера

```bash
# Проверка кода
ruff check .

# Автоматическое исправление
ruff check --fix .

# Форматирование
ruff format .
```

## Docker

### Запуск через Docker Compose
```bash
docker-compose up -d
```

Jupyter будет доступен по адресу: http://localhost:8888

## Структура результатов

После выполнения скриптов:
- `experiments/results.csv` - таблица с результатами экспериментов
- `experiments/visualizations.png` - визуализации данных
- `data/processed/best_model.pkl` - лучшая обученная модель

## Чекпоинт 1 (CP1)

Для CP1 выполнены следующие требования:

### Обработка и подготовка данных (13 баллов)
- ✅ Поиск и источник данных (Kaggle)
- ✅ Описание датасета
- ✅ Очистка данных (пропуски, дубли, выбросы)
- ✅ Feature engineering
- ✅ Визуализации
- ✅ Корректный сплит (train/val/test)
- ✅ Выбор и обоснование метрик

### Моделирование и эксперименты (7 баллов)
- ✅ Baseline модель (Linear Regression)
- ✅ 4-5 моделей (Ridge, Lasso, RF, GB, XGBoost, LightGBM)
- ✅ Таблица экспериментов
- ✅ Ансамбль моделей

### Качество кода и воспроизводимость (5 баллов)
- ✅ Чистая структура проекта
- ✅ Линтер (ruff)
- ✅ Fixed seed (RANDOM_SEED = 42)
- ✅ requirements.txt и pyproject.toml
- ✅ Docker и docker-compose
- ✅ README с описанием структуры
