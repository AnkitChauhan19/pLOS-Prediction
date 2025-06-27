import pandas as pd
import numpy as np

## Data Preprocessing ##

# Loading data
df1 = pd.read_csv("C:/Users/ankit/Desktop/Ankit/College Internship- EHR/Datasets/eicu-collaborative-research-database-2.0/EICU/apacheApsVar.csv")
df2 = pd.read_csv("C:/Users/ankit/Desktop/Ankit/College Internship- EHR/Datasets/eicu-collaborative-research-database-2.0/EICU/apachePredVar.csv")
df3 = pd.read_csv("C:/Users/ankit/Desktop/Ankit/College Internship- EHR/Datasets/eicu-collaborative-research-database-2.0/EICU/vitalAperiodic.csv")
df4 = pd.read_csv("C:/Users/ankit/Desktop/Ankit/College Internship- EHR/Datasets/eicu-collaborative-research-database-2.0/EICU/lab.csv")
df5 = pd.read_csv("C:/Users/ankit/Desktop/Ankit/College Internship- EHR/Datasets/eicu-collaborative-research-database-2.0/EICU/apachePatientResult.csv")

# Selecting necessary columns from the apacheApsVar table
apacheApsVar = df1.loc[:, [
    'patientunitstayid', 'temperature', 'heartrate', 'urine', 'wbc', 'bun', 'sodium', 'bilirubin'
]]

# Selecting necessary columns from the apachePredVar table
apachePredVar = df2.loc[:, [
    'patientunitstayid', 'age', 'aids', 'lymphoma', 'leukemia', 'metastaticcancer', 'electivesurgery', 'pao2', 'fio2',
    'verbal', 'motor', 'eyes'
]]
# Filling null values with 2 to classify null values as medical admission in ICU(neither elective nor non-elective)
apachePredVar['electivesurgery'].fillna(2, inplace=True)
# Selecting patients with age between 18 and 90
apachePredVar = apachePredVar[(apachePredVar['age'] >= 18) & (apachePredVar['age'] <= 90)]

# Selecting necessary columns from the apachePatientResult table
apachePatient = df5[df5['actualicumortality'] != 'EXPIRED'][['patientunitstayid', 'actualiculos']]
apachePatient = apachePatient.groupby('patientunitstayid').apply(lambda x: x.sample(1)).reset_index(drop=True)

# Selecting necessary columns from the vitalAperiodic table
vitals = df3.loc[:, ['patientunitstayid', 'observationoffset', 'noninvasivesystolic']]
# Selecting maximum and minimum values of systolic bp for first 24 hours after ICU admission for each patient
# Also dropping records with NaN values
vitals = vitals[(vitals['observationoffset'] >= 0) & (vitals['observationoffset'] <= 1440)].dropna()
vital1 = vitals.loc[vitals.groupby('patientunitstayid')['noninvasivesystolic'].idxmax()]
vital2 = vitals.loc[vitals.groupby('patientunitstayid')['noninvasivesystolic'].idxmin()]
# Renaming columns to maxsystolic and minsystolic to represent the extreme values
vital1 = vital1.reset_index(drop = True).rename(columns = {'noninvasivesystolic' : 'maxsystolic'})
vital2 = vital2.reset_index(drop = True).rename(columns = {'noninvasivesystolic' : 'minsystolic'})
# Merging the two extreme values of systolic bp of each patient into a single table
vitalAperiodic = pd.merge(vital1, vital2, on='patientunitstayid', how='inner')
vitalAperiodic = vitalAperiodic.loc[:, ['patientunitstayid', 'maxsystolic', 'minsystolic']]

# Selecting the records of potassium and bicarbonate level of patients from lab table
lab1 = df4[df4['labname'] == 'potassium'].rename(columns = {'labresult': 'potassium'})
lab2 = df4[df4['labname'] == 'bicarbonate'].rename(columns = {'labresult': 'bicarbonate'})
# Dropping records with NaN values
lab1 = lab1.loc[:, ['patientunitstayid', 'labresultrevisedoffset', 'potassium']].dropna()
lab2 = lab2.loc[:, ['patientunitstayid', 'labresultrevisedoffset', 'bicarbonate']].dropna()
# Selecting records for first 24 hours after ICU admission
lab1 = lab1[(lab1['labresultrevisedoffset'] >= 0) & (lab1['labresultrevisedoffset'] <= 1440)]
lab2 = lab2[(lab2['labresultrevisedoffset'] >= 0) & (lab2['labresultrevisedoffset'] <= 1440)]
# Selecting maximum and minimum values of potassium levels for each patient
lab1_max = lab1.loc[lab1.groupby('patientunitstayid')['potassium'].idxmax()]
lab1_min = lab1.loc[lab1.groupby('patientunitstayid')['potassium'].idxmin()]
lab1_max = lab1_max.reset_index(drop = True).rename(columns = {'potassium' : 'maxpotassium'})
lab1_min = lab1_min.reset_index(drop = True).rename(columns = {'potassium' : 'minpotassium'})
# Selecting maximum and minimum values of bicarbonate levels for each patient
lab2_max = lab2.loc[lab2.groupby('patientunitstayid')['bicarbonate'].idxmax()]
lab2_min = lab2.loc[lab2.groupby('patientunitstayid')['bicarbonate'].idxmin()]
lab2_max = lab2_max.reset_index(drop=True).rename(columns = {'bicarbonate' : 'maxbicarbonate'})
lab2_min = lab2_min.reset_index(drop=True).rename(columns = {'bicarbonate' : 'minbicarbonate'})
# Merging the potassium and bicarbonate levels data into a single table
lab1 = pd.merge(lab1_min, lab1_max, on=['patientunitstayid'], how='inner')
lab2 = pd.merge(lab2_min, lab2_max, on=['patientunitstayid'], how='inner')
lab = pd.merge(lab1, lab2, on=['patientunitstayid'], how='outer')
lab = lab.loc[:, ['patientunitstayid', 'maxpotassium', 'minpotassium', 'maxbicarbonate', 'minbicarbonate']]

# Merging all the columns from different tables to get a single dataset
final = pd.merge(apacheApsVar, apachePredVar, on='patientunitstayid', how='inner')
final = pd.merge(final, vitalAperiodic, on='patientunitstayid', how='inner')
final = pd.merge(final, lab, on='patientunitstayid', how='inner')
final = pd.merge(final, apachePatient, on='patientunitstayid', how='inner')

# Dropping records which have a missing ICU LOS value
final = final.dropna(subset = ['actualiculos'])
# Classifying the actual los(length of stay) data into two categories(prolonged if los > 3 days and not prolonged otherwise)
final['actualiculos'] = final['actualiculos'].apply(lambda x: 1 if x > 3 else 0)

# Replacing -ve values(not measured or recorded values) with NaN
numerical_cols = final.select_dtypes(include=['number']).columns
final[numerical_cols] = final[numerical_cols].apply(lambda col: col.mask(col < 0))

# Dropping records which have more than 30% NaN/NULL values
final = final.dropna(thresh = final.shape[1] - int(0.3 * final.shape[1]))

# Filling the NaN values for each column with the mean normal range values for that column
for col in numerical_cols:
    final[col].fillna(final[col].median(), inplace=True)

# Calculating the GCS(Glasgow comma score) value and storing it in dataset
final['verbal'] = final['verbal'] + final['motor'] + final['eyes']
final = final.drop(columns = ['motor', 'eyes'])
# Checking for hematological malignancy(lymphoma or leukemia) and storing output in dataset
final['lymphoma'] = final['lymphoma'] | final['leukemia']
final = final.drop(columns = ['leukemia'])
# Calculating the PaO2/FiO2 ratio and storing it in dataset
final['pao2'] = final['pao2'] / (final['fio2'] / 100)
final = final.drop(columns = ['fio2']).rename(columns = {'pao2' : 'pao2/fio2'})
final['pao2/fio2'] = final['pao2/fio2'].fillna(final['pao2/fio2'].median())

# Renaming the necessary columns
final = final.rename(columns = {'verbal': 'gcs', 'lymphoma' : 'hematological'})

# The complete dataset to be used for prediction model
dataset = final.drop(columns = ['patientunitstayid'])
print(dataset.shape[0])

# Saving dataset in a csv file
dataset.to_csv('C:/Users/ankit/Desktop/Ankit/College Internship- EHR/pLOS Prediction/pLOS_Dataset_new.csv', index = False)