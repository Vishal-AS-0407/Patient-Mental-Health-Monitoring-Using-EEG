import numpy as np
import pandas as pd
import joblib  

ada_boost_model = joblib.load('multi_label_ada_boost_model.pkl') 

X = pd.read_csv('Dreamer.csv').values  

sample = X[0].reshape(1, -1) 

pred = ada_boost_model.predict(sample)

print(f"Predicted Class for the 1st sample: {pred}")
