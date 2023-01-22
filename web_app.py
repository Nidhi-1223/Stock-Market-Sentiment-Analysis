import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pickle
import streamlit as st
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

# loading the saved model
loaded_model = pickle.load(open('models\LDA_model_improved.pkl', 'rb'))

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

# creating a function for Prediction
def stock_market_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The price will decrease'
    else:
      return 'The price will increase'

# -------------------for graph (data - upload_DIJA_table)
data = pd.read_csv('data/upload_DJIA_table.csv')
data['Date'] = pd.to_datetime(data['Date'])

#function to make chart 
def make_chart(data, y_col, ymin, ymax):
    fig = go.Figure(layout_yaxis_range=[ymin, ymax])
    fig.add_trace(go.Scatter(x=data['Date'], y=data[y_col], mode='lines+markers'))
    
    fig.update_layout(width=900, height=570, xaxis_title='time',
    yaxis_title=y_col)
    st.write(fig)

def main():
    # giving a title
    st.title('Stock Market Price Prediction')


    # getting the input data from the user
    open = st.text_input('Enter the opening price: ')
    high = st.text_input('Enter the highest price: ')
    low = st.text_input('Enter the lowest price: ')
    close = st.text_input('Enter the closing price: ')
    volume = st.text_input('Enter the volume of stocks: ')
    sample_news = st.text_input('Enter the news line: ')

    #preprocessing
    sample_news = re.sub(pattern='[^a-zA-Z]',repl=' ', string=sample_news)
    sample_news = sample_news.lower()

    subjectivity = getSubjectivity(sample_news)
    polarity = getPolarity(sample_news)
    SIA = getSIA(sample_news)

    compund=SIA['compound']
    neg=SIA['neg']
    neu=SIA['neu']
    pos=SIA['pos']

    # creating a button for Prediction
    if st.button('Predict'):
        result = stock_market_prediction([open,high,low,close,volume,subjectivity,polarity,compund,neg,pos,neu])
        st.success(result)
    

    ##-----------for graph-----------------------------
    #defining containers
    header = st.container()
    select_param = st.container()
    plot_spot = st.empty()
    #title
    with header:
        st.subheader("Daily Stock Prices")

    #select parmeter drop down
    with select_param:
        param_lst = list(data.columns)
        param_lst.remove('Date')
        select_param = st.selectbox('Select a Parameter',   param_lst)

    # calling the chart function
    n = len(data)
    ymax = max(data[select_param])+5
    ymin = min(data[select_param])-5
    for i in range(0, n-30, 1):
        df_tmp = data.iloc[i:i+30, :]
        with plot_spot:
            make_chart(df_tmp, select_param, ymin, ymax)
        time.sleep(0.5)
     
if __name__ == '__main__':
    main()