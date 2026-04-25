"""Visualization script for Twitch dataset."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent / "src"))

from config import RANDOM_SEED
from data_loader import load_twitch_data
from utils import set_seed

set_seed(RANDOM_SEED)
sns.set_style("whitegrid")


def main():
    """Create visualizations for Twitch dataset."""
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ДАННЫХ TWITCH")
    print("=" * 80)

    data_path = Path("data/raw/twitchdata.csv")
    if not data_path.exists():
        print(f"\n❌ Файл не найден: {data_path}")
        return

    df = load_twitch_data(data_path)
    print(f"\n📊 Загружено: {len(df)} строк")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Анализ данных Twitch стримеров", fontsize=16, fontweight="bold")

    # 1. Distribution of Watch time
    if "Watch time(Minutes)" in df.columns:
        ax = axes[0, 0]
        df["Watch time(Minutes)"].hist(bins=50, ax=ax, edgecolor="black")
        ax.set_title("Распределение Watch Time")
        ax.set_xlabel("Watch Time (Minutes)")
        ax.set_ylabel("Частота")
        ax.grid(True, alpha=0.3)

    # 2. Average viewers vs Followers
    if "Average viewers" in df.columns and "Followers" in df.columns:
        ax = axes[0, 1]
        ax.scatter(df["Followers"], df["Average viewers"], alpha=0.5)
        ax.set_title("Followers vs Average Viewers")
        ax.set_xlabel("Followers")
        ax.set_ylabel("Average Viewers")
        ax.grid(True, alpha=0.3)

    # 3. Top games
    if "Game" in df.columns:
        ax = axes[1, 0]
        top_games = df["Game"].value_counts().head(10)
        top_games.plot(kind="barh", ax=ax)
        ax.set_title("Топ-10 игр по количеству стримеров")
        ax.set_xlabel("Количество стримеров")
        ax.grid(True, alpha=0.3)

    # 4. Correlation heatmap
    ax = axes[1, 1]
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, center=0)
        ax.set_title("Корреляционная матрица")

    plt.tight_layout()
    
    # Save figure
    output_path = Path("experiments/visualizations.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ Визуализация сохранена: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
