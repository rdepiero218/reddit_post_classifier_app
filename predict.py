import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@st.cache ## decorator
def predict_input(classifier_model, input):
    return classifier_model.predict(input)