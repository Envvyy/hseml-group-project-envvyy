"""Training script for Twitch popularity prediction models."""

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE, VAL_SIZE
from data_loader import load_twitch_data
from features import create_engagement_features, encode_categorical_features, scale_features
from models import evaluate_model, get_baseline_model, get_models_dict, train_and_evaluate
from preprocessing import handle_missing_values, remove_duplicates, remove_outliers, split_data
from utils import save_experiment_results, set_seed

# Set seed
set_seed(RANDOM_SEED)


def prepare_data(df: pd.DataFrame, target_col: str):
    """Prepare data for modeling."""
    print("\n" + "=" * 80)
    print("ПОДГОТОВКА ДАННЫХ")
    print("=" * 80)

    # Remove duplicates
    print(f"\n1. Удаление дубликатов...")
    df_clean = remove_duplicates(df)
    print(f"   Удалено: {len(df) - len(df_clean)} строк")

    # Handle missing values
    print(f"\n2. Обработка пропущенных значений...")
    df_clean = handle_missing_values(df_clean, strategy="drop")
    print(f"   Осталось строк: {len(df_clean)}")

    # Feature engineering
    print(f"\n3. Feature engineering...")
    df_clean = create_engagement_features(df_clean)
    print(f"   Создано новых признаков")

    # Encode categorical
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print(f"\n4. Кодирование категориальных признаков...")
        df_clean, encoders = encode_categorical_features(df_clean, categorical_cols)
        print(f"   Закодировано: {len(categorical_cols)} признаков")

    # Remove outliers from target
    print(f"\n5. Удаление выбросов из целевой переменной...")
    df_clean = remove_outliers(df_clean, [target_col], method="iqr")
    print(f"   Осталось строк: {len(df_clean)}")

    return df_clean


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ - TWITCH POPULARITY PREDICTION")
    print("=" * 80)

    # Load data
    data_path = Path("data/raw/twitchdata.csv")
    if not data_path.exists():
        print(f"\n❌ Файл не найден: {data_path}")
        return

    print(f"\n📂 Загрузка данных...")
    df = load_twitch_data(data_path)
    print(f"   Загружено: {len(df)} строк, {len(df.columns)} столбцов")

    # Choose target
    target_col = "Watch time(Minutes)"
    print(f"\n🎯 Целевая переменная: {target_col}")
    print(f"   Обоснование: Watch time - ключевая метрика популярности стримера")
    print(f"   Отражает общее время просмотра и вовлеченность аудитории")

    # Prepare data
    df_clean = prepare_data(df, target_col)

    # Split features and target
    print("\n" + "=" * 80)
    print("РАЗДЕЛЕНИЕ ДАННЫХ")
    print("=" * 80)

    # Drop non-feature columns
    drop_cols = ["Channel", target_col]
    feature_cols = [col for col in df_clean.columns if col not in drop_cols]

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"\n📊 Признаков: {len(feature_cols)}")
    print(f"   Целевая переменная: {target_col}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
    )

    print(f"\n✂️  Разделение данных:")
    print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"\n   Избежание data leak: используем train_test_split с фиксированным seed")

    # Scale features
    print(f"\n📏 Масштабирование признаков...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_val, X_test
    )

    # Metrics explanation
    print("\n" + "=" * 80)
    print("МЕТРИКИ КАЧЕСТВА")
    print("=" * 80)
    print("\n📊 Выбранные метрики:")
    print("   1. RMSE (Root Mean Squared Error) - основная метрика")
    print("      Штрафует большие ошибки сильнее, важно для бизнеса")
    print("   2. MAE (Mean Absolute Error) - дополнительная")
    print("      Средняя абсолютная ошибка, более интерпретируема")
    print("   3. R² Score - коэффициент детерминации")
    print("      Показывает долю объясненной дисперсии")

    # Baseline model
    print("\n" + "=" * 80)
    print("BASELINE MODEL")
    print("=" * 80)
    print("\n🔹 Linear Regression (без feature engineering)")

    baseline = get_baseline_model()
    baseline_metrics = train_and_evaluate(
        baseline, X_train, y_train, X_val, y_val
    )

    print(f"\n   RMSE: {baseline_metrics['RMSE']:,.2f}")
    print(f"   MAE:  {baseline_metrics['MAE']:,.2f}")
    print(f"   R²:   {baseline_metrics['R2']:.4f}")

    # Save baseline results
    save_experiment_results(
        "baseline",
        "Linear Regression",
        baseline_metrics,
        {},
        "experiments/results.csv",
    )

    # Train multiple models
    print("\n" + "=" * 80)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("=" * 80)

    models = get_models_dict(RANDOM_SEED)
    results = []

    for name, model in models.items():
        print(f"\n🔸 {name}...")
        try:
            metrics = train_and_evaluate(
                model, X_train_scaled, y_train, X_val_scaled, y_val
            )
            print(f"   RMSE: {metrics['RMSE']:,.2f}")
            print(f"   MAE:  {metrics['MAE']:,.2f}")
            print(f"   R²:   {metrics['R2']:.4f}")

            results.append({"model": name, **metrics})

            # Save results
            save_experiment_results(
                "main_experiment",
                name,
                metrics,
                model.get_params() if hasattr(model, "get_params") else {},
                "experiments/results.csv",
            )
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")

    # Results table
    print("\n" + "=" * 80)
    print("ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("RMSE")
    print("\n" + results_df.to_string(index=False))

    # Best model
    best_model_name = results_df.iloc[0]["model"]
    best_rmse = results_df.iloc[0]["RMSE"]
    print(f"\n🏆 Лучшая модель: {best_model_name}")
    print(f"   RMSE: {best_rmse:,.2f}")

    # Train ensemble
    print("\n" + "=" * 80)
    print("АНСАМБЛЬ МОДЕЛЕЙ")
    print("=" * 80)
    print("\n🔸 Voting Regressor (RF + XGBoost + LightGBM)...")

    from sklearn.ensemble import RandomForestRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    ensemble = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
            ("lgbm", LGBMRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)),
        ]
    )

    ensemble_metrics = train_and_evaluate(
        ensemble, X_train_scaled, y_train, X_val_scaled, y_val
    )
    print(f"\n   RMSE: {ensemble_metrics['RMSE']:,.2f}")
    print(f"   MAE:  {ensemble_metrics['MAE']:,.2f}")
    print(f"   R²:   {ensemble_metrics['R2']:.4f}")

    # Save ensemble
    save_experiment_results(
        "ensemble",
        "Voting Regressor",
        ensemble_metrics,
        {},
        "experiments/results.csv",
    )

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА TEST SET")
    print("=" * 80)

    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    y_test_pred = best_model.predict(X_test_scaled)
    test_metrics = evaluate_model(y_test, y_test_pred)

    print(f"\n🏆 {best_model_name} на тестовой выборке:")
    print(f"   RMSE: {test_metrics['RMSE']:,.2f}")
    print(f"   MAE:  {test_metrics['MAE']:,.2f}")
    print(f"   R²:   {test_metrics['R2']:.4f}")

    # Save best model
    model_path = PROCESSED_DATA_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n💾 Модель сохранена: {model_path}")

    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
