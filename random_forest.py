
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
def clean_data(df):
    df.apply(lambda x: sum(x.isnull()),axis=0)

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)

def categorical(df):
    var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
        df.dtypes
df1=pd.read_csv("C:/Users/rogunda/Desktop/analytics_vidhya/train_ctrUa4K.csv")
clean_data(df1)
categorical(df1)
df1['LoanAmount_log'] = np.log(df1['LoanAmount'])
df1['TotalIncome'] = df1['ApplicantIncome'] + df1['CoapplicantIncome']
df1['TotalIncome_log'] = np.log(df1['TotalIncome'])
'''
model=SVC(kernel="rbf",random_state=0)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
X_train=df1[predictor_var].values
'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History']
X_train=df1[predictor_var].values
outcome_var='Loan_Status'
y_train=df1[outcome_var].values

model.fit(X_train,y_train)

df2=pd.read_csv("C:/Users/rogunda/Desktop/analytics_vidhya/test_lAUu6dG11.csv")
clean_data(df2)
categorical(df2)
df2['LoanAmount_log'] = np.log(df2['LoanAmount'])
df2['TotalIncome'] = df2['ApplicantIncome'] + df2['CoapplicantIncome']
df2['TotalIncome_log'] = np.log(df2['TotalIncome'])
X_test=df2[predictor_var].values
y_pred=list(model.predict(X_test))
