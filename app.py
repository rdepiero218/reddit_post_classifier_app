import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from predict import predict_input

startrek_url = 'https://www.reddit.com/r/startrek/'
fallout_url = 'https://www.reddit.com/r/Fallout/'
###------------------------

st.title('Subreddit Post Classifier')

st.header('Fallout vs Star Trek')

### dispaly logo images
col1, col2 = st.columns(2)
with col1:
    st.image('./images/fallout-logo.png', use_container_width=True)
with col2:
    st.image('./images/startrek-logo.png', use_container_width=True)

### explanation of app
st.subheader('What does it do?')
st.markdown('This app will take the text from a post in reddit and classify it as from either the Fallout or Star Trek subreddits. Note: this classifier is for text only!')

### load model
with open('saved_rforest_model.pkl', 'rb') as f:
    clf = pickle.load(f)

###------------------------

### user input
st.header('Try it Out!')
st.markdown('You can either enter your own sample text below or visit either subreddit to pull a post to test.')

col1, col2 = st.columns(2)

### fancy buttons don't work on heroku :(
with col1:
    st.link_button('Visit Fallout Subreddit', fallout_url)

with col2:
    st.link_button('Visit Star Trek Subreddit', startrek_url)


st.markdown('Once you have your sample post, enter the text below and hit the button to classify it.')
user_input = st.text_input('Enter Text Here', value='I love Fallout 4', label_visibility='collapsed')

button_clicked = st.button('Classify It!')


if button_clicked:
    test_post = pd.Series(str(user_input))
    ### make predictions
    prediction = predict_input(clf, test_post)
    st.header('Your post is from the subreddit for')
    if prediction == 1:
        st.image('./images/fallout-logo.png', use_column_width=True)
    else:
        st.image('./images/startrek-logo.png', use_column_width=True)
else:
    st.empty()










