import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def run_baseline():
    print("--- Запуск базовых моделей (Baseline) ---")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, 'sales_history.csv')

    # 2. Загрузка данных
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    except FileNotFoundError:
        print(f"Ошибка: Файл с данными не найден по пути:\n{DATA_PATH}")
        print("Сначала запустите скрипт генерации данных (data_generation.py)!")
        return

    items = df['item_id'].unique()
    all_naive_maes = []
    all_lr_maes = []

    for item in items:
        print(f"\n--- Анализ для товара: {item} ---")
        item_df = df[df['item_id'] == item].copy()
        item_df = item_df.sort_values('date').reset_index(drop=True)
        item_df['time_index'] = np.arange(len(item_df))

        split_index = int(len(item_df) * 0.8)
        train = item_df.iloc[:split_index]
        test = item_df.iloc[split_index:]

        print(f"Train: {len(train)} дней | Test: {len(test)} дней")

        # Наивный прогноз
        naive_predictions = item_df['sales_volume'].shift(1).iloc[split_index:]
        naive_mae = mean_absolute_error(test['sales_volume'], naive_predictions)
        all_naive_maes.append(naive_mae)
        print(f"Результат 1: MAE Наивного прогноза = {naive_mae:.2f} шт.")

        # Линейная регрессия
        X_train = train[['time_index']]
        y_train = train['sales_volume']
        X_test = test[['time_index']]
        y_test = test['sales_volume']

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_predictions = lr_model.predict(X_test)

        lr_mae = mean_absolute_error(y_test, lr_predictions)
        all_lr_maes.append(lr_mae)
        print(f"Результат 2: MAE Линейной регрессии = {lr_mae:.2f} шт.")

    print("\n==============================")
    print("--- ОБЩИЙ ИТОГ ПО ВСЕМ ТОВАРАМ ---")
    avg_naive = np.mean(all_naive_maes)
    avg_lr = np.mean(all_lr_maes)

    print(f"Среднее MAE Наивного прогноза: {avg_naive:.2f}")
    print(f"Среднее MAE Линейной регрессии: {avg_lr:.2f}")

    best_baseline = min(avg_naive, avg_lr)
    print(f"\n🎯 НАША ЦЕЛЬ ДЛЯ НЕЙРОСЕТИ:")
    print(f"Обучить модель, которая выдаст MAE строго меньше {best_baseline:.2f}!")

if __name__ == "__main__":
    run_baseline()