import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

gbdt_data = pd.read_csv('gbdt_prediction.csv')
data = pd.read_csv('pLOS_Dataset_new.csv')

X = gbdt_data.drop(columns = ['actualiculos'])
y = gbdt_data['actualiculos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gbdt = GradientBoostingClassifier(
    n_estimators = 200, 
    min_samples_split = 400, 
    max_depth = 6, 
    max_features = 'sqrt', 
    subsample = 0.7, 
    random_state = 42
)

rf = RandomForestClassifier(
    n_estimators = 200,
    min_samples_split = 200,
    max_features = 'sqrt',
    random_state = 42
)

dl = MLPClassifier(
    hidden_layer_sizes = (150, 150),
    solver = 'adam',
    alpha = 0.5,
    random_state = 42,
    max_iter = 500
)

gbdt.fit(X_train, y_train)
rf.fit(X_train, y_train)
dl.fit(X_train_scaled, y_train)

y_pred_prob_gbdt = gbdt.predict_proba(X_test)[:, 1]
y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_prob_dl = dl.predict_proba(X_test_scaled)[:, 1]

## Analysing the hybrid model using AUROC, AUPRC and Brier score ##

auroc = roc_auc_score(y_test, y_pred_prob_gbdt)
print(f'AUROC for GBDT-GBDT is: {auroc}')
auprc = average_precision_score(y_test, y_pred_prob_gbdt)
print(f'AUPRC for GBDT-GBDT is: {auprc}')
brier_score = brier_score_loss(y_test, y_pred_prob_gbdt)
print(f'Brier Score for GBDT-GBDT is: {brier_score}')


auroc_rf = roc_auc_score(y_test, y_pred_prob_rf)
print(f'AUROC for GBDT-RF is: {auroc_rf}')
auprc_rf = average_precision_score(y_test, y_pred_prob_rf)
print(f'AUPRC for GBDT-RF is: {auprc_rf}')
brier_score_rf = brier_score_loss(y_test, y_pred_prob_rf)
print(f'Brier Score for GBDT-RF is: {brier_score_rf}')

auroc_dl = roc_auc_score(y_test, y_pred_prob_dl)
print(f'AUROC for GBDT-DL is: {auroc_dl}')
auprc_dl = average_precision_score(y_test, y_pred_prob_dl)
print(f'AUPRC for GBDT-DL is: {auprc_dl}')
brier_score_dl = brier_score_loss(y_test, y_pred_prob_dl)
print(f'Brier Score for GBDT-DL is: {brier_score_dl}')

# AUROC for GBDT-GBDT is: 0.7833737029896818
# AUPRC for GBDT-GBDT is: 0.5949454173796784
# Brier Score for GBDT-GBDT is: 0.1404320179244553

# AUROC for GBDT-RF is: 0.779984873833089
# AUPRC for GBDT-RF is: 0.5893349162945194
# Brier Score for GBDT-RF is: 0.14317600285884446

# AUROC for GBDT-DL is: 0.7774344519578609
# AUPRC for GBDT-DL is: 0.5797352570915447
# Brier Score for GBDT-DL is: 0.14339857407331724