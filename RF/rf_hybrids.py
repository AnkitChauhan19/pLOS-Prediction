import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

rf_data = pd.read_csv('rf_prediction.csv')

X = rf_data.drop(columns = ['actualiculos'])
y = rf_data['actualiculos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True, random_state = 42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gbdt = GradientBoostingClassifier(
    n_estimators = 200, 
    min_samples_split = 300, 
    max_depth = 6, 
    max_features = 'sqrt', 
    subsample = 0.7, 
    random_state = 42
)

rf = RandomForestClassifier(
    n_estimators = 200,
    min_samples_split = 100,
    max_features = 'sqrt',
    random_state = 42
)

dl = MLPClassifier(
    hidden_layer_sizes = (100, 100),
    solver = 'adam',
    alpha = 0.5,
    random_state = 42,
    max_iter = 500
)

gbdt.fit(X_train, y_train)
# rf.fit(X_train, y_train)
# dl.fit(X_train_scaled, y_train)

y_pred_prob_gbdt = gbdt.predict_proba(X_test)[:, 1]
# y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]
# y_pred_prob_dl = dl.predict_proba(X_test_scaled)[:, 1]

## Analysing the hybrid model using AUROC, AUPRC and Brier score ##
gbdt_importances = gbdt.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gbdt_importances
})
feature_importance['Importance'] = feature_importance['Importance'] * 100
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)
# auroc = roc_auc_score(y_test, y_pred_prob_gbdt)
# print(f'AUROC for RF-GBDT is: {auroc}')
# auprc = average_precision_score(y_test, y_pred_prob_gbdt)
# print(f'AUPRC for RF-GBDT is: {auprc}')
# brier_score = brier_score_loss(y_test, y_pred_prob_gbdt)
# print(f'Brier Score for RF-GBDT is: {brier_score}')

# auroc_rf = roc_auc_score(y_test, y_pred_prob_rf)
# print(f'AUROC for RF-RF is: {auroc_rf}')
# auprc_rf = average_precision_score(y_test, y_pred_prob_rf)
# print(f'AUPRC for RF-RF is: {auprc_rf}')
# brier_score_rf = brier_score_loss(y_test, y_pred_prob_rf)
# print(f'Brier Score for RF-RF is: {brier_score_rf}')

# auroc_dl = roc_auc_score(y_test, y_pred_prob_dl)
# print(f'AUROC for RF-DL is: {auroc_dl}')
# auprc_dl = average_precision_score(y_test, y_pred_prob_dl)
# print(f'AUPRC for RF-DL is: {auprc_dl}')
# brier_score_dl = brier_score_loss(y_test, y_pred_prob_dl)
# print(f'Brier Score for RF-DL is: {brier_score_dl}')

# AUROC for RF-GBDT is: 0.7798233538022167
# AUPRC for RF-GBDT is: 0.5857583367682104
# Brier Score for RF-GBDT is: 0.1425414606933563

# AUROC for RF-RF is: 0.7782640777958847
# AUPRC for RF-RF is: 0.5860203773244348
# Brier Score for RF-RF is: 0.14343190263394903

# AUROC for RF-DL is: 0.7742207703126548
# AUPRC for RF-DL is: 0.5704192078606308
# Brier Score for RF-DL is: 0.1448375044994622