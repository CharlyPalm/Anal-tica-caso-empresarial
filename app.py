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

#Función para cargar los datos ya limpios del archivo CSV
def load_data():
    print("Cargando datos...")
    df = pd.read_csv("last_time_of_day.csv")
    print("Los datos se cargaron con éxito:", df.shape)
    
    # Preprocesamiento de los datos
    df['date_only'] = pd.to_datetime(df['date_only'])  # Convert date_only to datetime
    df['date_ordinal'] = df['date_only'].map(dt.datetime.toordinal)  # Convert date_only to ordinal
    df.set_index('date_only', inplace=True)  # Set date_only as index
    
    return df

#Función para entrenar el modelo con los datos cargados
def train_model(data):
    X = data[['date_ordinal', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]
    Y = data['close']
    
    # Split de los datos para entrenamiento y prueba, 80% y 20% respectivamente
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    
    # Calcular el error cuadrático medio y el coeficiente de determinación
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    
    return model, mse, r2

# Función para predecir el precio de Bitcoin en una fecha específica
def predict_price(model, input_date):
    input_date_ordinal = pd.Timestamp(input_date).toordinal()
    X_new = pd.DataFrame([[input_date_ordinal, 0, 0, 0, 0, 0]], columns=['date_ordinal', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'])
    predicted_price = model.predict(X_new)
    return predicted_price[0]

# Función para hacer plot de un gráfico de velas, extraído de https://www.kaggle.com/code/kaanxtr/btc-usdt-visualization-hourly-4-hourly-daily
def plot_candlestick_chart(dataframe, title):
    fig = go.Figure(data=[go.Candlestick(x=dataframe.index,
                                         open=dataframe['open'],
                                         high=dataframe['high'],
                                         low=dataframe['low'],
                                         close=dataframe['close'])])

    fig.update_layout(title=title,
                      xaxis_title='Fecha',
                      yaxis_title='Precio (USDT)',
                      xaxis_rangeslider_visible=False)

    return fig

# Función que extrae noticias al dashboard
def fetch_news(api_key, query="stocks", language="en", page_size=8):
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&pageSize={page_size}&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error("No se pudieron cargar las noticias. Inténtalo de nuevo más tarde.")
        return []

# Función para mostrar las noticias en el dashboard
def display_news(api_key):
    st.header("📈 Noticias del mercado de valores")
    news_articles = fetch_news(api_key)
    for article in news_articles:
        st.subheader(article['title'])
        st.write(article['description'])
        st.write(f"[Read more]({article['url']})")
        st.image(article['urlToImage'], use_column_width=True)

# Main
def main():
    data = load_data()
    model, mse, r2 = train_model(data)
    
    st.title('Precio de Bitcoin y calculadora de inversión')
    
    # Display candlestick chart with axis labels
    fig = plot_candlestick_chart(data, 'BTC/USDT Price History')
    st.plotly_chart(fig)

    # Sidebar for about and user inputs
    st.sidebar.title("About")
    st.sidebar.info('Esta aplicación muestra los datos históricos de precios de Bitcoin desde un archivo CSV y predice precios futuros.\n\n')
    st.sidebar.title("Entrenar el modelo")
    if st.sidebar.button("Entrenar el modelo"):
        model, mse, r2 = train_model(data)
        st.write("El modelo ha sido entrenado con éxito!")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R^2 Score: {r2}")
        # Impresión de la gráfica de regresión con los valores reales y predichos
        st.write("Gráfico de regresión")
        plt.scatter(model.predict(data[['date_ordinal', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]), data['close'], color='black')
        plt.title('Regresión Lineal')
        plt.xlabel('Precio Real')
        plt.ylabel('Precio Predicho')
        plt.show()
        st.pyplot()

        
    # User Inputs
    st.sidebar.title("Menú")
    investment_date = st.sidebar.date_input("Seleccione una fecha de inversion")
    withdrawal_date = st.sidebar.date_input("Seleccione una fecha de retiro")
    investment_amount = st.sidebar.number_input("Ingrese la cantidad a invertir", min_value=0.0, step=10.0)

    if st.sidebar.button("Invertir!!"):
        try:
            if 'model' in locals():
                investment_price = round(predict_price(model, investment_date),2)
                withdrawal_price = round(predict_price(model, withdrawal_date),2)
                st.write(f"Precio previsto para la fecha de inversión: ({investment_date}): {investment_price}")
                st.write(f"Precio previsto para la fecha de retiro ({withdrawal_date}): {withdrawal_price}")
                
                profit_or_loss = (withdrawal_price - investment_price) * investment_amount
                if profit_or_loss > 0:
                    st.success(f"Ganarás ${profit_or_loss:.2f} si te retiras en {withdrawal_date}.")
                    st.balloons()
                else:
                    st.error(f"Perderás ${-profit_or_loss:.2f} si te retiras en {withdrawal_date}.")
            else:
                st.write("Por favor, entrena el modelo primero")
        except ValueError:
            st.sidebar.error("Ingresa fechas válidas")
    
    if st.sidebar.button("Noticias!!"):
        st.title("Dashboard de noticias del mercado de valores")
        # Add a section for recent news
        api_key = "a19623fc2eb149859e6096bc0714fa93"  
        display_news(api_key)

        #


if __name__ == '__main__':
    main()

