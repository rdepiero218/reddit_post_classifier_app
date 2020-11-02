
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import predict
import pickle

### read in data
data = pd.read_csv('./data/reddit_posts_clean.csv')

### select data
X = data['selftext']
y = data['is_fallout']
### TTS
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


### construct model
model = make_pipeline(CountVectorizer(stop_words='english'), RandomForestClassifier())
### fit model
model.fit(X_train, y_train)

### save fitted model in pickle file
with open('saved_rforest_model.pkl', 'wb') as file:
    pickle.dump(model, file)
