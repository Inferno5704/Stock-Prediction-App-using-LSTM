# 📈 Stock Price Prediction using LSTM

A powerful web application built with Streamlit that uses Long Short-Term Memory (LSTM) neural networks to predict stock prices. The app features real-time data fetching, hyperparameter tuning, and interactive visualizations.

## 🌐 Live Demo

**Try the app now:** [Stock Prediction App](https://stock-prediction-app-using-lstm-srcekst9mu5trhppbacbck.streamlit.app/)

No installation required - just click the link above to start predicting stock prices!

## 🌟 Features

- **Real-time Stock Data**: Fetches live stock data using Yahoo Finance API
- **LSTM Neural Networks**: Advanced deep learning model for time series prediction
- **Hyperparameter Tuning**: Automatic optimization of model parameters
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Multiple Metrics**: Comprehensive model evaluation (MAE, MSE, RMSE, R²)
- **Future Predictions**: Forecasts stock prices for specified future periods
- **Popular Tickers**: Pre-loaded list of popular stocks including US and Indian markets
- **Customizable Parameters**: Adjustable sequence length, prediction periods, and model settings

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Inferno5704/Stock-Prediction-App-using-LSTM.git
   cd Stock-Prediction-App-using-LSTM
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   The app will automatically open at `http://localhost:8501`

## 📊 How to Use

### 1. Configure Settings
- **Stock Selection**: Choose from popular tickers or enter a custom ticker symbol
- **Date Range**: Set training start date and train/test split date
- **Sequence Length**: Adjust the number of previous days used for prediction (30-100)
- **Future Days**: Set how many days ahead to predict (10-90)

### 2. Model Parameters
- **Hyperparameter Tuning**: Enable automatic optimization (recommended)
- **Manual Configuration**: Choose specific units, batch size, epochs, optimizer, and activation function

### 3. Run Prediction
Click the "🚀 Start Prediction" button to:
- Load and visualize historical data
- Train the LSTM model
- Display evaluation metrics
- Show actual vs predicted prices
- Generate future price forecasts

## 🔧 Technical Details

### Model Architecture
- **Input Layer**: LSTM with configurable units and tanh/relu activation
- **Hidden Layer**: Second LSTM layer for deeper learning
- **Output Layer**: Dense layer for final price prediction
- **Optimizer**: Adam, RMSprop, or SGD
- **Loss Function**: Mean Squared Error

### Supported Tickers
**US Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN, META, NFLX, NVDA
**Indian Stocks**: RELIANCE.NS, TCS.NS, INFY.NS
**Custom**: Any valid Yahoo Finance ticker symbol

### Evaluation Metrics
- **MAE**: Mean Absolute Error - average absolute difference
- **MSE**: Mean Squared Error - average squared difference
- **RMSE**: Root Mean Squared Error - standard deviation of residuals
- **R² Score**: Coefficient of determination - model accuracy percentage

## 📁 Project Structure

```
stock-lstm-prediction/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file (recommended)
```

## ⚙️ Dependencies

- **streamlit**: Web app framework
- **yfinance**: Yahoo Finance data fetching
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **scikit-learn**: Machine learning utilities
- **tensorflow**: Deep learning framework

## 🎯 Use Cases

- **Individual Investors**: Make informed investment decisions
- **Financial Analysts**: Analyze stock trends and patterns
- **Students**: Learn about time series prediction and LSTM networks
- **Researchers**: Experiment with different model configurations
- **Day Traders**: Short-term price movement analysis

## 📈 Performance Tips

1. **Data Quality**: Use stocks with sufficient historical data (2+ years recommended)
2. **Sequence Length**: Longer sequences capture more patterns but require more computation
3. **Hyperparameter Tuning**: Enable for better accuracy but longer training time
4. **Market Conditions**: Model performance varies with market volatility

## ⚠️ Disclaimers

- **Not Financial Advice**: This tool is for educational and research purposes only
- **Past Performance**: Historical data doesn't guarantee future results
- **Market Risk**: Stock markets are inherently unpredictable
- **Data Dependency**: Predictions are only as good as the input data quality

## 🙏 Acknowledgments

Special thanks to **[supremeapollo](https://github.com/supremeapollo)** for their valuable collaboration and contributions to this project.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Useful Links

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [LSTM Networks Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the [Streamlit Community Forum](https://discuss.streamlit.io/)
- Review the documentation links above

---

⭐ **Star this repository if you found it helpful!** ⭐
