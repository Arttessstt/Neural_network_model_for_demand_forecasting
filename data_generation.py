import pandas as pd
import numpy as np
import os

# Фиксируем seed для стабильности
np.random.seed(42)

dates = pd.date_range(start='2022-01-01', periods=730, freq='D')
t = np.arange(len(dates)) 

def generate_item_data(item_id, base_price, base_sales, season_amp, trend_amp, noise_scale):
    # 1. Годовая сезонность (плавная волна 365 дней)
    seasonality = season_amp * np.sin(2 * np.pi * t / 365)
    
    # 2. Двухгодовой тренд
    trend = trend_amp * np.cos(2 * np.pi * t / 730)
    
    # 3. НОВОЕ: Ярко выраженная НЕДЕЛЬНАЯ сезонность (пики каждые 7 дней)
    weekly = 20 * np.cos(2 * np.pi * t / 7) 
    
    # 4. Небольшой случайный шум
    noise = np.random.normal(loc=0, scale=noise_scale, size=len(t))

    # Собираем всё вместе
    sales_volume = base_sales + seasonality + trend + weekly + noise
    sales_volume = np.maximum(0, np.round(sales_volume))

    prices = np.round(np.random.normal(loc=base_price, scale=base_price * 0.02, size=len(t)), 2)

    df = pd.DataFrame({
        'date': dates,
        'item_id': item_id,
        'price': prices,
        'sales_volume': sales_volume.astype(int)
    })
    return df

df_item_1 = generate_item_data('item_1', 150.0, 50, 25, 10, 5)
df_item_2 = generate_item_data('item_2', 450.0, 15, -10, 5, 3)

df = pd.concat([df_item_1, df_item_2])

# --- ЖЕЛЕЗОБЕТОННЫЕ ПУТИ ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = os.path.join(BASE_DIR, 'sales_history.csv')

df.to_csv(SAVE_PATH, index=False)
print(f"Файл успешно обновлен и сохранен по пути: {SAVE_PATH}")