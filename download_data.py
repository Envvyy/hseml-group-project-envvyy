"""Script to download Twitch dataset from Kaggle."""

import os
import sys

print("=" * 80)
print("ИНСТРУКЦИЯ ПО ЗАГРУЗКЕ ДАТАСЕТА")
print("=" * 80)
print("\n1. Перейдите на Kaggle: https://www.kaggle.com/datasets/aayushmishra1512/twitchdata")
print("2. Нажмите 'Download' и скачайте файл")
print("3. Распакуйте CSV файл в папку: data/raw/")
print("4. Переименуйте файл в 'twitchdata.csv' если нужно")
print("\nИЛИ используйте Kaggle API:")
print("   kaggle datasets download -d aayushmishra1512/twitchdata -p data/raw --unzip")
print("\n" + "=" * 80)
