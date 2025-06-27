import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
import pickle

dataset = pd.read_csv('pLOS_Dataset_new.csv')

## Model Creation ##

# Specifying the regressor variables - X and target variable - y
X = dataset.drop(columns = ['actualiculos'])
y = dataset['actualiculos']
# Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Prediction models specification
rf = RandomForestClassifier(
    n_estimators = 100,
    max_features = 'sqrt',
    min_samples_split = 2,
    random_state = 42
)

dl = MLPClassifier(
    hidden_layer_sizes = (100, 100),
    solver = 'adam',
    alpha = 1,
    random_state = 42,
    max_iter = 500
)

## Fitting data ##

rf.fit(X_train, y_train)
dl.fit(X_train_scaled, y_train)

## Predicting Values ##

y_pred_rf = rf.predict(X_test)
y_pred_dl = dl.predict(X_test_scaled)
y_pred_rf = rf.predict_proba(X_test)[:, 1]
y_pred_dl = dl.predict_proba(X_test_scaled)[:, 1]

## Saving models using pickle ##

with open('rf_model.pkl', 'wb') as rf_file:
    pickle.dump(rf, rf_file)

with open('dl_model.pkl', 'wb') as dl_file:
    pickle.dump(dl, dl_file)

## Saving new datasets with predicted values ##

train_data = pd.DataFrame(X_train)
train_data['actualiculos'] = y_train

test_data_rf = pd.DataFrame(X_test)
test_data_rf['actualiculos'] = y_pred_rf

test_data_dl = pd.DataFrame(X_test)
test_data_dl['actualiculos'] = y_pred_dl

combined_data_rf = pd.concat([train_data, test_data_rf], ignore_index=True)
combined_data_dl = pd.concat([train_data, test_data_dl], ignore_index=True)

combined_data_rf.to_csv('rf_prediction.csv', index=False)
combined_data_dl.to_csv('dl_prediction.csv', index=False)

## Analysing the model using AUROC, AUPRC and Brier score ##\

auroc_rf = roc_auc_score(y_test, y_pred_rf)
print(f'AUROC for RF is: {auroc_rf}')
auroc_dl = roc_auc_score(y_test, y_pred_dl)
print(f'AUROC for DL is: {auroc_dl}')

auprc_rf = average_precision_score(y_test, y_pred_rf)
print(f'AUPRC for RF is: {auprc_rf}')
auprc_dl = average_precision_score(y_test, y_pred_dl)
print(f'AUPRC for DL is: {auprc_dl}')

brier_score_rf = brier_score_loss(y_test, y_pred_rf)
print(f'Brier Score for RF is: {brier_score_rf}')
brier_score_dl = brier_score_loss(y_test, y_pred_dl)
print(f'Brier Score for DL is: {brier_score_dl}')

# AUROC for RF is: 0.732013823887823
# AUPRC for RF is: 0.5443664838347911
# Brier Score for RF is: 0.17546400783252225

# AUROC for DL is: 0.7313848456256371
# AUPRC for DL is: 0.545926228272732
# Brier Score for DL is: 0.17565988366674715