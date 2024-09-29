import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@st.cache_resource ## decorator
def predict_input(_classifier_model, input):
    return _classifier_model.predict(input)