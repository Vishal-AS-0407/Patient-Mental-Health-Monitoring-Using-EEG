import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
import joblib


X = pd.read_csv('Dreamer.csv').values  
y = pd.read_csv('DreamerTarget.csv').values  

X_shape = X.shape
y_shape = y.shape
print(f"Feature matrix shape: {X_shape}")
print(f"Target matrix shape: {y_shape}")


ada_boost_cls = AdaBoostClassifier( 
    n_estimators=165,                
    learning_rate=0.5,    
    algorithm='SAMME.R'           
)

multi_output_classifier = MultiOutputClassifier(ada_boost_cls)

multi_output_classifier.fit(X, y)

preds = multi_output_classifier.predict(X)

accuracy = accuracy_score(y, preds)
print(f"Final Accuracy: {accuracy:.4f}")

joblib.dump(multi_output_classifier, 'multi_label_ada_boost_model.pkl')


print(f"Model saved at: multi_label_ada_boost_model.pkl")
