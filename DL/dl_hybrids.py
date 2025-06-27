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
    max_depth = 4, 
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
    hidden_layer_sizes = (100, 100),
    solver = 'adam',
    alpha = 0.5,
    random_state = 42,
    max_iter = 500
)

# gbdt.fit(X_train, y_train)
# rf.fit(X_train, y_train)
dl.fit(X_train_scaled, y_train)

# y_pred_prob_gbdt = gbdt.predict_proba(X_test)[:, 1]
# y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_prob_dl = dl.predict_proba(X_test_scaled)[:, 1]

## Analysing the hybrid model using AUROC, AUPRC and Brier score ##

# auroc = roc_auc_score(y_test, y_pred_prob_gbdt)
# print(f'AUROC for DL-GBDT is: {auroc}')
# auprc = average_precision_score(y_test, y_pred_prob_gbdt)
# print(f'AUPRC for DL-GBDT is: {auprc}')
# brier_score = brier_score_loss(y_test, y_pred_prob_gbdt)
# print(f'Brier Score for DL-GBDT is: {brier_score}')

# auroc_rf = roc_auc_score(y_test, y_pred_prob_rf)
# print(f'AUROC for DL-RF is: {auroc_rf}')
# auprc_rf = average_precision_score(y_test, y_pred_prob_rf)
# print(f'AUPRC for DL-RF is: {auprc_rf}')
# brier_score_rf = brier_score_loss(y_test, y_pred_prob_rf)
# print(f'Brier Score for DL-RF is: {brier_score_rf}')

# auroc_dl = roc_auc_score(y_test, y_pred_prob_dl)
# print(f'AUROC for DL-DL is: {auroc_dl}')
# auprc_dl = average_precision_score(y_test, y_pred_prob_dl)
# print(f'AUPRC for DL-DL is: {auprc_dl}')
# brier_score_dl = brier_score_loss(y_test, y_pred_prob_dl)
# print(f'Brier Score for DL-DL is: {brier_score_dl}')
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob_dl, n_bins=10)
eci = np.mean(np.abs(prob_true - prob_pred))
print(f"Estimated Calibration Index (ECI): {eci:.4f}")
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('DL-DL')
plt.legend()
plt.show()
# AUROC for DL-GBDT is: 0.7740436972908271
# AUPRC for DL-GBDT is: 0.5743985097731985
# Brier Score for DL-GBDT is: 0.14152541649773395

# AUROC for DL-RF is: 0.7722158308409233
# AUPRC for DL-RF is: 0.5724294334238373
# Brier Score for DL-RF is: 0.14331716842394457

# AUROC for DL-DL is: 0.7719288978381947
# AUPRC for DL-DL is: 0.573022205901256
# Brier Score for DL-DL is: 0.1415928237481559