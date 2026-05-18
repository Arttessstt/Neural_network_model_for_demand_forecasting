import streamlit as st
import pandas as pd
import torch
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import datetime
import pickle
import os

def apply_custom_design():
    st.markdown("""
        <style>
        [data-testid="stMetric"], div[data-testid="metric-container"] {
            background-color: #1e212b;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #ffd700;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
            transition: transform 0.2s;
            color: white;
        }
        [data-testid="stMetric"]:hover { transform: translateY(-5px); }
        .stButton>button {
            width: 100%; background-color: #ffd700 !important; color: black !important;
            font-weight: bold !important; border: none !important; padding: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

# 1. АРХИТЕКТУРА (Строго совпадает с train_nn.py)
class DemandModel(torch.nn.Module):
    def __init__(self, input_size):
        super(DemandModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))

# 2. ЖЕСТКАЯ ЗАГРУЗКА БЕЗ КЭША (Защита от ошибки NoneType)
def load_model(model_path, input_size=7):
    if os.path.exists(model_path):
        model = DemandModel(input_size=input_size)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    return None

def load_scaler(scaler_path):
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

def show_theory_section():
    st.markdown("### Метод скользящего окна (Sliding Window)")
    st.write("Нейросеть смотрит на последние 7 дней как на 'фотографию' и предсказывает следующий день. Для прогноза на неделю вперед она использует свои же ответы как новые данные!")
    cols = st.columns(8)
    for i in range(7):
        cols[i].markdown(f"<div style='text-align:center; border: 1px solid #ffd700; padding:5px;'>День {i+1}<br><b>Вход</b></div>", unsafe_allow_html=True)
    cols[7].markdown(f"<div style='text-align:center; background:#00ff0033; border: 1px solid #00ff00; padding:5px;'>Цель<br><b>Выход</b></div>", unsafe_allow_html=True)

# 3. ОСНОВНАЯ ЛОГИКА
def main():
    apply_custom_design()
    st.title("Аналитика спроса")
    
    WINDOW_SIZE = 7
    
    # --- ЖЕЛЕЗОБЕТОННЫЕ ПУТИ ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model_weights.pt')
    SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
    
    model = load_model(MODEL_PATH, WINDOW_SIZE)
    scaler = load_scaler(SCALER_PATH)
    
    st.sidebar.header("📁 Загрузка данных")
    uploaded_file = st.sidebar.file_uploader("Выберите CSV-файл:", type=['csv'])

    if uploaded_file is not None:
        if not model or not scaler:
            st.error("⚠️ Модель или скейлер не найдены! Сначала запустите `train_nn.py` для их создания.")
            st.stop()

        df_raw = pd.read_csv(uploaded_file)
        
        with st.sidebar:
            st.success("Данные успешно загружены!")
            st.header("⚙️ Маппинг колонок")
            date_col = st.selectbox("Дата", df_raw.columns, index=0)
            item_col = st.selectbox("Товар/Категория", df_raw.columns, index=1)
            qty_col = st.selectbox("Кол-во продаж", df_raw.columns, index=3 if len(df_raw.columns) > 3 else len(df_raw.columns)-1)
            
            st.divider()
            st.header("🏢 Складские параметры")
            price = st.number_input("Цена (₽)", value=150)
            stock_now = st.slider("Текущий остаток", 0, 2000, 150)

            st.divider()
            st.header("📈 Настройки AI")
            forecast_days = st.slider("Горизонт прогноза (дней)", 1, 14, 7)

        # Предобработка
        try:
            df = df_raw.copy()
            df['date'] = pd.to_datetime(df[date_col])
            df['item_id'] = df[item_col]
            df['sales_volume'] = pd.to_numeric(df[qty_col], errors='coerce')
            df = df.dropna(subset=['date', 'item_id', 'sales_volume'])
            df = df.groupby(['date', 'item_id'], as_index=False)[['sales_volume']].sum()
            df = df.sort_values(['item_id', 'date'])
        except Exception as e:
            st.error(f"Ошибка формата данных: {e}")
            st.stop()

        target_item = st.selectbox("Выберите артикул для анализа:", df['item_id'].unique())
        item_df = df[df['item_id'] == target_item].copy()
        
        if len(item_df) <= WINDOW_SIZE:
            st.warning(f"Недостаточно данных для {target_item} (нужно > {WINDOW_SIZE} дней)")
            st.stop()

        # Рекурсивный прогноз
        scaled_vals = scaler.transform(item_df[['sales_volume']])
        current_window = scaled_vals[-WINDOW_SIZE:].flatten().tolist()
        future_predictions = []

        with torch.no_grad():
            for _ in range(forecast_days):
                input_tensor = torch.FloatTensor(current_window[-WINDOW_SIZE:]).view(1, -1)
                pred_scaled = model(input_tensor).item()
                
                current_window.append(pred_scaled)
                pred_real = int(max(0, scaler.inverse_transform([[pred_scaled]])[0][0]))
                future_predictions.append(pred_real)

        # Метрики
        actual_last = int(item_df['sales_volume'].iloc[-1])
        total_forecast = sum(future_predictions)
        is_risk = total_forecast > stock_now
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Прогноз ({forecast_days} дн.)", f"{total_forecast} шт")
        c2.metric("Выручка (ожид.)", f"{total_forecast * price:,} ₽")
        c3.metric("Запас", f"{stock_now} шт", f"-{total_forecast} потребность", delta_color="inverse" if is_risk else "normal")
        c4.metric("Статус", "Дефицит 🛑" if is_risk else "В норме ✅")

        # График
        hist_tail = item_df.tail(30)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_tail['date'], y=hist_tail['sales_volume'], name="Фактические", line=dict(color='#ffd700', width=3), mode='lines+markers'))
        
        last_date = hist_tail['date'].max()
        future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        plot_dates = [last_date] + future_dates
        plot_preds = [actual_last] + future_predictions

        fig.add_trace(go.Scatter(x=plot_dates, y=plot_preds, name=f"AI Прогноз", line=dict(color='#00ff00', width=3, dash='dash'), mode='lines+markers'))
        
        fig.add_vrect(
            x0=last_date, x1=future_dates[-1],
            fillcolor="rgba(0,255,0,0.05)", layer="below", line_width=0,
            annotation_text="Зона прогноза", annotation_position="top left"
        )
        fig.update_layout(template="plotly_dark", hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🎓 Как нейросеть делает этот прогноз? (Теория)"):
            show_theory_section()

    else:
        st.info("👋 Добро пожаловать! Загрузите CSV-файл с продажами в боковой панели слева, чтобы начать.")
        show_theory_section()

if __name__ == "__main__":
    main()