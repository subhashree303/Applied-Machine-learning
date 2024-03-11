import sklearn
import numpy as np
import pandas as pd
import pickle
from typing import Tuple
from sklearn.pipeline import Pipeline

with open(r"D:/cmi/sem 4/AppliedML/assi3/finetunedlogistic.pkl", 'rb') as file:
    best_model = pickle.load(file)



def score(text:str, model:Pipeline, threshold=0.5) -> Tuple[bool , float]:
   
    tfidf_text = model.named_steps['tfidf'].transform([text])
    propensity = model.named_steps['classifier'].predict_proba(tfidf_text)[:, 1][0]
    prediction = (propensity >= threshold)
    
    return bool(prediction), propensity