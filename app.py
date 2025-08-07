import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
warnings.filterwarnings('ignore')

# --- Streamlit config ---
st.set_page_config(page_title="Stock LSTM Forecast", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# --- Sidebar ---
st.sidebar.header("Configuration")

popular_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA", "RELIANCE.NS", "TCS.NS", "INFY.NS"]
selected = st.sidebar.selectbox("Choose Popular Ticker", popular_tickers)
custom = st.sidebar.text_input("Or enter custom ticker", value="")
ticker = custom.strip().upper() if custom else selected

start_date = pd.to_datetime(st.sidebar.date_input("Training Start Date", value=pd.to_datetime("2020-01-01")))
split_date = pd.to_datetime(st.sidebar.date_input("Train/Test Split Date", value=pd.to_datetime("2025-01-01")))
seq_len = st.sidebar.slider("Sequence Length", 30, 100, 60)
future_days = st.sidebar.slider("Future Prediction Days", 10, 90, 30)

# --- Model Parameters ---
st.sidebar.subheader("Model Parameters")
enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=True)

if not enable_tuning:
    combo = st.sidebar.selectbox(
        "Choose Configuration (Units, Batch Size, Epochs)",
        options=[
            (50, 32, 10),
            (50, 64, 20),
            (64, 32, 10),
            (64, 64, 20),
            (100, 32, 30)
        ],
        format_func=lambda x: f"Units: {x[0]}, Batch: {x[1]}, Epochs: {x[2]}"
    )
    units, batch_size, epochs = combo
    optimizer = st.sidebar.selectbox("Optimizer", ["adam", "rmsprop", "sgd"], index=0)
    activation = st.sidebar.selectbox("Activation Function", ["tanh", "relu"], index=0)
else:
    optimizer = "adam"
    activation = "tanh"

# --- Functions ---
@st.cache_data
def load_data(ticker, start_date):
    df = yf.download(ticker, start=start_date)
    if df.empty:
        st.error("No data found.")
        return None
    return df[['Close']]

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_lstm_model(X_train, y_train, units, batch_size, epochs):
    model = Sequential([
        LSTM(units, activation=activation, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units, activation=activation),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# --- Main ---
if st.button("🚀 Start Prediction", type="primary"):
    df = load_data(ticker, start_date)
    if df is not None:
        df = df[df.index >= start_date]
        current_price = float(df['Close'].iloc[-1])

        st.metric("Current Price", f"${current_price:.2f}")
        st.line_chart(df['Close'])

        df_train = df[df.index < split_date]
        df_test = df[df.index >= split_date]

        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(df_train)
        scaled_test = scaler.transform(df_test)

        X_train, y_train = create_sequences(scaled_train, seq_len)
        combined = np.concatenate((scaled_train[-seq_len:], scaled_test), axis=0)
        X_test, y_test = create_sequences(combined, seq_len)

        if enable_tuning:
            best_loss = float('inf')
            best_model = None
            results_data = []

            with st.spinner("Tuning hyperparameters..."):
                for u in [50, 64]:
                    for b in [32, 64]:
                        for e in [10, 20]:
                            st.write(f"Testing: Units={u}, Batch={b}, Epochs={e}")
                            model = train_lstm_model(X_train, y_train, u, b, e)
                            loss = model.evaluate(X_test, y_test, verbose=0)
                            results_data.append({
                                'Units': u,
                                'Batch Size': b,
                                'Epochs': e,
                                'Test Loss': loss
                            })
                            if loss < best_loss:
                                best_loss = loss
                                best_model = model

            results_df = pd.DataFrame(results_data)
            st.subheader("🔍 Hyperparameter Tuning Results")
            st.dataframe(results_df.sort_values('Test Loss'))
            st.success(f"Best Loss: {best_loss:.6f}")
        else:
            best_model = train_lstm_model(X_train, y_train, units, batch_size, epochs)

        predictions = best_model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
        # Compute metrics
        mae = mean_absolute_error(actual_prices, predicted_prices)
        mse = mean_squared_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_prices, predicted_prices)

        # Display in Streamlit
        st.subheader("📊 Model Evaluation Metrics")
        st.write(f"**MAE (Mean Absolute Error):** {mae:.4f}")
        st.write(f"**MSE (Mean Squared Error):** {mse:.4f}")
        st.write(f"**RMSE (Root Mean Squared Error):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        st.line_chart(pd.DataFrame({"Actual": actual_prices.flatten(), "Predicted": predicted_prices.flatten()}, index=df_test.index[:len(predicted_prices)]))

        last_seq = X_test[-1]
        future_preds = []
        cur_seq = last_seq.copy()
        for _ in range(future_days):
            pred = best_model.predict(cur_seq.reshape(1, seq_len, 1), verbose=0)[0][0]
            future_preds.append(pred)
            cur_seq = np.append(cur_seq[1:], [[pred]], axis=0)

        future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)

        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_prices.flatten()})
        st.line_chart(forecast_df.set_index("Date"))

        predicted_price = future_prices[-1][0]
        change = predicted_price - current_price
        pct = (change / current_price) * 100

        st.metric("Predicted Price", f"${predicted_price:.2f}", f"{pct:+.2f}%")
