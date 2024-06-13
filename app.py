import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime as dt
import plotly.graph_objects as go
import requests

def load_data():
    print("Loading data...")
    df = pd.read_csv("data/raw/last_time_of_day.csv")
    print("Data loaded successfully:", df.shape)
    
    # Preprocess data if needed
    df['date_only'] = pd.to_datetime(df['date_only'])  # Convert date_only to datetime
    df['date_ordinal'] = df['date_only'].map(dt.datetime.toordinal)  # Convert date_only to ordinal
    df.set_index('date_only', inplace=True)  # Set date_only as index
    
    return df

def train_model(data):
    X = data[['date_ordinal', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]
    Y = data['close']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    
    return model, mse, r2

def predict_price(model, input_date):
    input_date_ordinal = pd.Timestamp(input_date).toordinal()
    X_new = pd.DataFrame([[input_date_ordinal, 0, 0, 0, 0, 0]], columns=['date_ordinal', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'])
    predicted_price = model.predict(X_new)
    return predicted_price[0]

def plot_candlestick_chart(dataframe, title):
    fig = go.Figure(data=[go.Candlestick(x=dataframe.index,
                                         open=dataframe['open'],
                                         high=dataframe['high'],
                                         low=dataframe['low'],
                                         close=dataframe['close'])])

    fig.update_layout(title=title,
                      xaxis_title='Date',
                      yaxis_title='Price (USDT)',
                      xaxis_rangeslider_visible=False)

    return fig

def fetch_news(api_key, query="stocks", language="en", page_size=8):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error("Failed to fetch news")
        return []

def display_news(api_key):
    st.header("📈 Stock Market News")
    news_articles = fetch_news(api_key)
    for article in news_articles:
        st.subheader(article['title'])
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
        st.image(article['urlToImage'], use_column_width=True)


def main():
    data = load_data()
    model, mse, r2 = train_model(data)
    
    st.title('Bitcoin Price History and Prediction')
    
    # Display candlestick chart with axis labels
    fig = plot_candlestick_chart(data, 'BTC/USDT Price History')
    st.plotly_chart(fig)

    # Sidebar for about and user inputs
    st.sidebar.title("About")
    st.sidebar.info('This app displays the historical price data of Bitcoin from a CSV file and predicts future prices.\n\n')

    st.sidebar.title("Model Training")
    if st.sidebar.button("Train Model"):
        model, mse, r2 = train_model(data)
        st.write("Model trained successfully!")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R^2 Score: {r2}")
    
    # User Inputs
    st.sidebar.title("User Inputs")
    investment_date = st.sidebar.date_input("Select an Investment Date")
    withdrawal_date = st.sidebar.date_input("Select a Withdrawal Date")
    investment_amount = st.sidebar.number_input("Enter the Amount to Invest", min_value=0.0, step=100.0)

    if st.sidebar.button("Invest!!"):
        try:
            if 'model' in locals():
                investment_price = predict_price(model, investment_date)
                withdrawal_price = predict_price(model, withdrawal_date)
                st.write(f"Predicted price for investment date ({investment_date}): {investment_price}")
                st.write(f"Predicted price for withdrawal date ({withdrawal_date}): {withdrawal_price}")
                
                profit_or_loss = (withdrawal_price - investment_price) * investment_amount
                if profit_or_loss > 0:
                    st.success(f"You will gain ${profit_or_loss:.2f} if you withdraw on {withdrawal_date}.")
                    st.balloons()
                else:
                    st.error(f"You will lose ${-profit_or_loss:.2f} if you withdraw on {withdrawal_date}.")
            else:
                st.write("Please train the model first.")
        except ValueError:
            st.sidebar.error("Please enter valid dates.")
    
    if st.sidebar.button("News!!"):
        st.title("Stock Trading Dashboard")
        # Add a section for recent news
        api_key = "a19623fc2eb149859e6096bc0714fa93"  
        display_news(api_key)

        #


if __name__ == '__main__':
    main()

