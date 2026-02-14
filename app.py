from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

# Load all saved objects
le = pickle.load(open('./p_models/label_encoder.pkl', 'rb'))
sc = pickle.load(open('./p_models/standard_scaler.pkl', 'rb'))
ohe = pickle.load(open('./p_models/onehot_encoder.pkl', 'rb'))
oe = pickle.load(open('./p_models/ordinal_encoder.pkl', 'rb'))
rf = pickle.load(open('./p_models/churn_model.pkl', 'rb'))
feature_names = pickle.load(open('./p_models/feature_names.pkl', 'rb'))

print(feature_names)