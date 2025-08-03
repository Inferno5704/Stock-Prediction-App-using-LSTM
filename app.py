# import streamlit as st
# import yfinance as yf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import warnings
# warnings.filterwarnings('ignore')

# # Page configuration
# st.set_page_config(
#     page_title="Stock Price Prediction with LSTM",
#     page_icon="📈",
#     layout="wide"
# )

# st.title("📈 Stock Price Prediction using LSTM")
# st.markdown("This app uses LSTM neural networks to predict stock prices based on historical data.")

# # Sidebar for user inputs
# st.sidebar.header("Configuration")
# ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")
# start_date = st.sidebar.date_input("Training Start Date", value=pd.to_datetime("2020-01-01"))
# split_date = st.sidebar.date_input("Train/Test Split Date", value=pd.to_datetime("2025-01-01"))
# seq_len = st.sidebar.slider("Sequence Length", min_value=30, max_value=100, value=60, help="Number of past days to use for prediction")
# future_days = st.sidebar.slider("Future Prediction Days", min_value=10, max_value=90, value=30)

# # Advanced settings
# st.sidebar.subheader("Model Parameters")
# enable_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=True)
# if not enable_tuning:
#     units = st.sidebar.selectbox("LSTM Units", [32, 50, 64, 100], index=1)
#     batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
#     epochs = st.sidebar.selectbox("Epochs", [10, 20, 30, 50], index=1)

# # Function to load and prepare data
# @st.cache_data
# def load_data(ticker, start_date):
#     try:
#         data = yf.download(ticker, start=start_date)
#         if data.empty:
#             st.error(f"No data found for ticker {ticker}")
#             return None
#         return data[['Close']].copy()
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return None

# # Function to create sequences
# def create_sequences(data, seq_len):
#     X, y = [], []
#     for i in range(seq_len, len(data)):
#         X.append(data[i-seq_len:i])
#         y.append(data[i])
#     return np.array(X), np.array(y)

# # Function to train LSTM model
# def train_lstm_model(X_train, y_train, units=50, batch_size=32, epochs=10):
#     model = Sequential([
#         LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
#         LSTM(units),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
    
#     # Create progress bar
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     class CustomCallback:
#         def __init__(self, total_epochs):
#             self.total_epochs = total_epochs
            
#         def on_epoch_end(self, epoch, logs=None):
#             progress = (epoch + 1) / self.total_epochs
#             progress_bar.progress(progress)
#             status_text.text(f'Training Progress: Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs.get("loss", 0):.6f}')
    
#     # Custom training loop for progress tracking
#     history = []
#     for epoch in range(epochs):
#         hist = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
#         history.append(hist.history['loss'][0])
#         progress = (epoch + 1) / epochs
#         progress_bar.progress(progress)
#         status_text.text(f'Training Progress: Epoch {epoch + 1}/{epochs} - Loss: {hist.history["loss"][0]:.6f}')
    
#     progress_bar.empty()
#     status_text.empty()
    
#     return model

# # Main app logic
# if st.button("🚀 Start Prediction", type="primary"):
#     # Load data
#     with st.spinner("Loading stock data..."):
#         df = load_data(ticker, start_date)
    
#     if df is not None:
#         # Display data info
#         st.subheader("📊 Data Overview")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total Records", len(df))
#         with col2:
#             st.metric("Date Range", f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
#         with col3:
#             st.metric("Current Price", f"${df['Close'].iloc[-1].item():.2f}")

        
#         # Plot historical data
#         st.subheader("📈 Historical Stock Price")
#         fig, ax = plt.subplots(figsize=(12, 6))
#         ax.plot(df.index, df['Close'], linewidth=1)
#         ax.axvline(x=split_date, color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
#         ax.set_title(f"{ticker} Historical Stock Price")
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Price ($)")
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         st.pyplot(fig)
#         plt.close()
        
#         # Prepare data
#         with st.spinner("Preparing data..."):
#             # Split data
#             df_train = df[df.index < pd.to_datetime(split_date)]
#             df_test = df[df.index >= pd.to_datetime(split_date)]
            
#             if len(df_train) < seq_len or len(df_test) == 0:
#                 st.error("Insufficient data for training/testing. Please adjust your date range.")
#                 st.stop()
            
#             # Scale data
#             scaler = MinMaxScaler()
#             scaled_train = scaler.fit_transform(df_train)
#             scaled_test = scaler.transform(df_test)
            
#             # Create sequences
#             X_train, y_train = create_sequences(scaled_train, seq_len)
            
#             # Prepare test sequences
#             combined = np.concatenate((scaled_train[-seq_len:], scaled_test), axis=0)
#             X_test, y_test = create_sequences(combined, seq_len)
        
#         st.success(f"Data prepared successfully! Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
#         # Model training
#         st.subheader("🤖 Model Training")
        
#         if enable_tuning:
#             st.info("Performing hyperparameter tuning...")
#             best_loss = float('inf')
#             best_model = None
#             best_params = {}
            
#             param_combinations = [
#                 (50, 32, 10), (50, 32, 20), (50, 64, 10), (50, 64, 20),
#                 (64, 32, 10), (64, 32, 20), (64, 64, 10), (64, 64, 20)
#             ]
            
#             results_data = []
            
#             for i, (units, batch_size, epochs) in enumerate(param_combinations):
#                 st.write(f"Testing combination {i+1}/{len(param_combinations)}: Units={units}, Batch={batch_size}, Epochs={epochs}")
                
#                 model = train_lstm_model(X_train, y_train, units, batch_size, epochs)
#                 loss = model.evaluate(X_test, y_test, verbose=0)
                
#                 results_data.append({
#                     'Units': units,
#                     'Batch Size': batch_size,
#                     'Epochs': epochs,
#                     'Test Loss': loss
#                 })
                
#                 st.write(f"Loss: {loss:.6f}")
                
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_model = model
#                     best_params = {'units': units, 'batch_size': batch_size, 'epochs': epochs}
            
#             # Display results table
#             results_df = pd.DataFrame(results_data)
#             st.subheader("🏆 Hyperparameter Tuning Results")
#             st.dataframe(results_df.sort_values('Test Loss'))
#             st.success(f"Best parameters: {best_params} with loss: {best_loss:.6f}")
            
#         else:
#             st.info("Training model with specified parameters...")
#             best_model = train_lstm_model(X_train, y_train, units, batch_size, epochs)
#             best_loss = best_model.evaluate(X_test, y_test, verbose=0)
#             st.success(f"Model trained with loss: {best_loss:.6f}")
        
#         # Make predictions
#         st.subheader("📊 Model Predictions")
        
#         with st.spinner("Generating predictions..."):
#             predictions = best_model.predict(X_test)
#             predicted_prices = scaler.inverse_transform(predictions)
#             actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
        
#         # Calculate metrics
#         mse = np.mean((actual_prices - predicted_prices) ** 2)
#         rmse = np.sqrt(mse)
#         mae = np.mean(np.abs(actual_prices - predicted_prices))
#         mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        
#         # Display metrics
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("RMSE", f"{rmse:.2f}")
#         with col2:
#             st.metric("MAE", f"{mae:.2f}")
#         with col3:
#             st.metric("MAPE", f"{mape:.2f}%")
#         with col4:
#             st.metric("Test Loss", f"{best_loss:.6f}")
        
#         # Plot predictions vs actual
#         fig, ax = plt.subplots(figsize=(12, 6))
#         test_dates = df_test.index[:len(actual_prices)]
#         ax.plot(test_dates, actual_prices, label='Actual Prices', linewidth=2, alpha=0.8)
#         ax.plot(test_dates, predicted_prices, label='Predicted Prices', linewidth=2, alpha=0.8)
#         ax.set_title(f"{ticker} Stock Price Prediction (Test Period)")
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Price ($)")
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         st.pyplot(fig)
#         plt.close()
        
#         # Future predictions
#         st.subheader("🔮 Future Price Forecast")
        
#         with st.spinner("Generating future predictions..."):
#             last_seq = X_test[-1]
#             future_preds = []
#             cur_seq = last_seq.copy()
            
#             for _ in range(future_days):
#                 pred = best_model.predict(cur_seq.reshape(1, seq_len, 1), verbose=0)[0][0]
#                 future_preds.append(pred)
#                 cur_seq = np.append(cur_seq[1:], [[pred]], axis=0)
            
#             future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        
#         # Create future dates
#         last_date = df.index[-1]
#         future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
        
#         # Plot forecast
#         fig, ax = plt.subplots(figsize=(12, 6))
        
#         # Historical data (last 60 days)
#         recent_data = df.tail(60)
#         ax.plot(recent_data.index, recent_data['Close'], label='Historical', linewidth=2, color='blue')
        
#         # Future predictions
#         ax.plot(future_dates, future_prices, label=f'Forecast ({future_days} days)', linewidth=2, color='red', linestyle='--')
        
#         ax.set_title(f"{ticker} - Future Price Forecast")
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Price ($)")
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         st.pyplot(fig)
#         plt.close()
        
#         # Future predictions summary
#         current_price = current_price = df['Close'].iloc[-1].item()

#         predicted_price = future_prices[-1][0]
#         price_change = predicted_price - current_price
#         price_change_pct = (price_change / current_price) * 100
        
#         st.subheader("📋 Forecast Summary")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Current Price", f"${current_price:.2f}")
#         with col2:
#             st.metric(f"Predicted Price ({future_days} days)", f"${predicted_price:.2f}", f"{price_change:+.2f}")
#         with col3:
#             st.metric("Expected Change", f"{price_change_pct:+.2f}%")
        
#         # Download predictions
#         forecast_df = pd.DataFrame({
#             'Date': future_dates,
#             'Predicted_Price': future_prices.flatten()
#         })
        
#         csv = forecast_df.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Forecast Data",
#             data=csv,
#             file_name=f"{ticker}_forecast_{future_days}days.csv",
#             mime="text/csv"
#         )

# # Disclaimer
# st.sidebar.markdown("---")
# st.sidebar.markdown("""
# ⚠️ **Disclaimer**: This app is for educational purposes only. 
# Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions.
# """)

# # Footer
# st.markdown("---")
# st.markdown("Built with Streamlit • Powered by TensorFlow & yfinance")
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
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