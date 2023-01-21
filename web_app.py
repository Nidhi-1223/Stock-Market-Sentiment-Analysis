import numpy as np
import pickle
import streamlit as st
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# loading the saved model
loaded_model = pickle.load(open('models\LDA_saved_model.pkl', 'rb'))

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
#     x=[open,high,low,close,volume,subjectivity,polarity,compund,neg,pos,neu]

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

    # x=[open,high,low,close,volume,subjectivity,polarity,compund,neg,pos,neu]
    # pred=loaded_model.predict([x])[0]
    # if pred == 0:
    #     print('The prices will decrease')
    # else:
    #     print('The prices will increase')

    # creating a button for Prediction
    if st.button('Predict'):
        result = stock_market_prediction([open,high,low,close,volume,subjectivity,polarity,compund,neg,pos,neu])
        st.success(result)
     
if __name__ == '__main__':
    main()