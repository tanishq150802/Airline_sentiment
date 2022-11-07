import re
import string
import nltk
import numpy as np
import pickle
from fastapi import FastAPI
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
 
# load model
model = load_model('tf_better.h5')

stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']
STOPWORDS = set(stopwordlist)
def cleaning_mentions(text):
    return re.sub("@[A-Za-z0-9_]+","", text)
def cleaning_non_alpha(text):
    return re.sub("[^a-z0-9]"," ", text)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
def cleaning_URLs(data): #cleaning URLs (if any)
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ', data)
english_punctuations = string.punctuation #removing punctuations
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
def cleaning_repeating_char(text): #removing repeating characters
    return re.sub(r'(.)1+', r'1', text)
def cleaning_numbers(data): #removing numbers
    return re.sub('[0-9]+', '', data)
st = nltk.PorterStemmer() #stemming is generally more suitable for sentiment analysis problems
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class inference():
  def __init__(self,exp):
    self.exp=exp
  def process(self,exp):
    exp=cleaning_mentions(exp)
    exp=cleaning_non_alpha(exp)
    exp=cleaning_numbers(exp)
    exp=cleaning_URLs(exp)
    exp=cleaning_punctuations(exp)
    exp=cleaning_repeating_char(exp)
    exp=cleaning_numbers(exp)
    exp=stemming_on_text(exp)
    exp = tokenizer.texts_to_sequences([exp])
    exp = pad_sequences(exp, padding='post', maxlen=24)
    pred=model.predict(exp)
    if(pred[0][0]>=0.5):
      return ["Positive",pred]
    else:
      return ["Negative",pred]

app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}


@app.get('/sentiment_analysis/')
async def query_sentiment_analysis(text: str):
    return analyze_sentiment(text)


def analyze_sentiment(text):
    """Get and process result"""

    result = inference(text)
    result=result.process(text)

    return result[0]

#/sentiment_analysis/?text="welcome"
# exp="I hate dancing!!"
# obj2=inference(exp)
# print(obj2.process(exp))