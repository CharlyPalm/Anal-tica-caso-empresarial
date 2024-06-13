import pandas as pd
import streamlit as st

@st.cache
def load_data():
    # Load the CSV file with specified columns
    df = pd.read_csv("../data/raw/BTCUSDT.csv")
    df = df[['timestamp', 'close', 'volume', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]
    
    # Ensure the data has the right format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    return df

def main():
    data = load_data()
    
    st.title('Currency Data')
    
    # Display the entire data as a line chart
    st.subheader('Price History')
    st.line_chart(data['close'])

    st.sidebar.title("About")
    st.sidebar.info('This app is a simple example of using Streamlit to create a financial data web app.\n\nIt is maintained by [Paduel](https://twitter.com/paduel_py).\n\nCheck the code at https://github.com/paduel/streamlit_finance_chart')

if _name_ == '_main_':
    main()
