import nltk
import random
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
import string
from tqdm import tqdm

def clean(df):
    whitelist=set(string.ascii_letters+' ')
    ingredients=[]
    for i,recipe in enumerate(df['ingredients']):
        foodlist=[]
        for food in recipe:
            food = food.lower()
            food = ''.join(filter(whitelist.__contains__,food))
            food = food.strip()
            food = food.split(" ")
            if('oz' in food):
                food.remove('oz')
            food = '_'.join(food)
            foodlist.append(food)
        if('oz' in foodlist):
            foodlist.remove('oz')
        ingredients.append(foodlist)
    df['ingredients']=ingredients
    return df
    
def data_process(df):
    stemmer=WordNetLemmatizer()

    ingredients=[]
    for index,item in enumerate(df['ingredients']):
        eg = []
        for i,d in enumerate(item):
            g=d.split("_")
            for ii,x in enumerate(g):
                g[ii]=stemmer.lemmatize(x)
            temp="_".join(g)
            eg.append(temp)
        str1 = ' '.join(eg)
        ingredients.append(str1)

    df['ingredients'] = pd.Series(ingredients)
    return df
    
def data_preprocess(train_dir):
    temp_df = pd.read_json(train_dir)
    temp_df=clean(temp_df)
    temp_df = data_process(temp_df)
    return temp_df