# This file contains the final model for predicting the persistency of a drug
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

df_drug = pd.read_excel("Healthcare_dataset.xlsx", 'Dataset')

# Let's remove the first character from the ID column for easier usage

df_drug['Ptid'] = df_drug['Ptid'].str[1:]

# Let's also remove the columns with only 1 unique value
df_drug = df_drug.drop(['Ptid', 'Risk_Immobilization', 'Risk_Estrogen_Deficiency', 'Risk_Chronic_Liver_Disease',
                       'Risk_Untreated_Early_Menopause', 'Risk_Untreated_Chronic_Hyperthyroidism', 'Risk_Osteogenesis_Imperfecta'], axis=1)
# Split into X and Y
drug_y = df_drug["Persistency_Flag"]
drug_x = df_drug.drop("Persistency_Flag", axis=1)


features = drug_x.columns
# Encode the columns
x_factorized = pd.DataFrame()
for feature in features:
    x_factorized[feature] = pd.factorize(drug_x[feature])[0]

drug_y = pd.DataFrame(pd.factorize(drug_y)[0])
drug_x = x_factorized
drug_y = np.ravel(drug_y)


X_train, X_test, y_train, y_test = train_test_split(
    drug_x, drug_y, test_size=0.3, random_state=42)

# Create classifier object
clf = RandomForestClassifier(n_estimators=100)

# Train the model
clf.fit(X_train, y_train)

# Dump to pickle file
pickle.dump(clf, open("model.pkl", "wb"))
