# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 18:03:21 2021

@author: ADMIN
"""
import joblib
import sklearn


#filename of Gradient Boosting model
joblib_file='grad_boost_tuned.pkl'
# Load the pickled model
grad_boost_model= joblib.load(joblib_file)

