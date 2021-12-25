# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 19:21:45 2021

@author: ADMIN
"""
import pandas as pd

#1)seat_comfort
#2)cabin_service
#3)food_bev
#4)entertainment
#5)ground_sesrvice
x_=["Business","Couple_Leisure","Family_Leisure","Solo_Leisure"]
y_=["Business Class","Economy Class","First Class","Premium Economy"]


def encdoding_data(a,b):
    features=[]
    travelling_type=x_.copy()
    cabin_=y_.copy()
    encode_1=pd.DataFrame(data={'travel':travelling_type})
    encode_2=pd.DataFrame(data={'cabin':cabin_})
    featu_1=pd.get_dummies(encode_1['travel'],drop_first=True).iloc[travelling_type.index(a),:].values
    featu_2=pd.get_dummies(encode_2['cabin'],drop_first=True).iloc[cabin_.index(b),:].values
    
    a,b=list(featu_1),list(featu_2)

    features.extend(a)
    features.extend(b)
    return features
    
    



