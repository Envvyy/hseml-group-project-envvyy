# Прогнозирование популярности стримеров Twitch

## Описание проекта
Проект направлен на предсказание популярности стримеров Twitch (по просмотрам или подписчикам) на основе характеристик канала: игры, частота стримов, средний онлайн и другие метрики.

## Структура проекта
```
project_twitch/
├── data/                   # Данные
│   ├── raw/               # Исходные данные
│   └── processed/         # Обработанные данные
├── notebooks/             # Jupyter notebooks для анализа
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Предобработка данных
│   └── 03_modeling.ipynb # Моделирование
├── src/                   # Исходный код
│   ├── __init__.py
│   ├── data_loader.py    # Загрузка данных
│   ├── preprocessing.py  # Предобработка
│   ├── features.py       # Feature engineering
│   ├── models.py         # Модели
│   └── utils.py          # Утилиты
├── experiments/           # Результаты экспериментов
├── .gitignore
├── requirements.txt       # Зависимости
├── pyproject.toml        # Конфигурация проекта
├── docker-compose.yml    # Docker конфигурация
├── Dockerfile
└── README.md
```

## Датасет
Источник: [Kaggle - Twitch Data](https://www.kaggle.com/datasets/aayushmishra1512/twitchdata)

## Установка

### Локально
```bash
pip install -r requirements.txt
```

### Docker
```bash
docker-compose up -d
```

## Использование
```bash
# Запуск Jupyter
jupyter notebook notebooks/
```

## Метрики качества
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

## Модели
- Baseline: Linear Regression
- Random Forest
- XGBoost
- LightGBM
- Ансамбли

## Автор
Проект выполнен в рамках курса по машинному обучению.
