import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve
import shap
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('pLOS_Dataset_new.csv')

## Model Construction ##

# Specifying the regressor variables - X and target variable - y
X = dataset.drop(columns = ['actualiculos'])
y = dataset['actualiculos']

# Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# GBDT prediction model specification
gbdt = GradientBoostingClassifier(
    subsample = 0.7,
    n_estimators = 100,
    max_features = 'sqrt',
    max_depth = 6,
    min_samples_split = 200,
    random_state = 42
)

# Parameter grid for selecting best parameters for the prediction model
# param_grid = {
#     'n_estimators' : [100, 200, 300],
#     'max_depth' : [6, 8, 10],
#     'min_samples_split' : [100, 200, 300],
#     'max_features' : ['sqrt', 'log2']
# }
# Using GridSearchCV model to select the best parameters for the prediction model
# grid_search = GridSearchCV(
#     estimator = gbdt,
#     param_grid = param_grid,
#     cv = 5,
#     scoring = 'roc_auc',
#     n_jobs = -1,
#     verbose = 2
# )

## Model Fitting ##

# grid_search.fit(X_train, y_train)
# # Values of best parameters and models
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_
# print(best_params)

gbdt.fit(X_train, y_train)

## Saving the model using pickle ##

with open('gbdt_model.pkl', 'wb') as gbdt_file:
    pickle.dump(gbdt, gbdt_file)

## Predicting Values ##

y_pred = gbdt.predict(X_test)[:, 1]
y_pred_proba = gbdt.predict_proba(X_test)[:, 1]

## Saving new dataset with predicted values ##

train_data = pd.DataFrame(X_train)
train_data['actualiculos'] = y_train

test_data = pd.DataFrame(X_test)
test_data['actualiculos'] = y_pred

combined_data = pd.concat([train_data, test_data], ignore_index=True)

combined_data.to_csv('gbdt_prediction.csv', index=False)

## Analysing the model using AUROC, AUPRC, Brier score and ECI ##

auroc = roc_auc_score(y_test, y_pred_proba)
print(f'AUROC test is: {auroc}')

auprc = average_precision_score(y_test, y_pred_proba)
print(f'AUPRC: {auprc}')

brier_score = brier_score_loss(y_test, y_pred_proba)
print(f'Brier Score: {brier_score}')

# Calibration curve for the GBDT model
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
eci = np.mean(np.abs(prob_true - prob_pred))
print(f"Estimated Calibration Index (ECI): {eci:.4f}")
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.show()

# Using feature_importances_ of gbdt classifier to know importance of each feature
gbdt_importances = gbdt.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': gbdt_importances
})
feature_importance['Importance'] = feature_importance['Importance'] * 100
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(feature_importance)

# Using SHapley Additive exPlanations(SHAP) to know importance of each feature
explainer = shap.TreeExplainer(gbdt)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names = X_test.columns)

# # AUROC test is: 0.741833725100244
# # AUPRC: 0.5606504039195837
# # Brier Score: 0.17227070460404353
# # Estimated Calibration Index (ECI): 0.0110

# #               Feature  Importance
# # 13               gcs   21.183220
# # 12         pao2/fio2   18.640292
# # 1          heartrate    9.417285
# # 15       minsystolic    9.403558
# # 4                bun    6.972693
# # 0        temperature    5.100358
# # 3                wbc    5.059036
# # 7                age    3.201314
# # 5             sodium    3.008333
# # 14       maxsystolic    2.778200
# # 6          bilirubin    2.578460
# # 17      minpotassium    2.533403
# # 2              urine    2.265414
# # 19    minbicarbonate    2.255329
# # 18    maxbicarbonate    1.899173
# # 16      maxpotassium    1.882758
# # 11   electivesurgery    1.592634
# # 10  metastaticcancer    0.127852
# # 9      hematological    0.086230
# # 8               aids    0.014458