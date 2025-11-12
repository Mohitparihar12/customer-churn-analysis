
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not os.path.exists(data_path):
    print("Please place the Telco Customer Churn CSV as:", data_path)
else:
    df = pd.read_csv(data_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df = df.drop(columns=['customerID'])
    df['Churn'] = df['Churn'].map({'No':0,'Yes':1})
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for c in cat_cols:
        if df[c].nunique() <= 2:
            df[c] = le.fit_transform(df[c])
        else:
            df = pd.get_dummies(df, columns=[c], prefix=[c], drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    num_cols = X_train.select_dtypes(include=['float64','int64']).columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
    plt.figure(figsize=(6,8))
    feat_imp.plot(kind='barh')
    plt.title('Top features')
    plt.savefig('churn_feature_importance.png', bbox_inches='tight')
    joblib.dump(rf, 'rf_churn_model.pkl')
    print('Artifacts saved: rf_churn_model.pkl, churn_feature_importance.png, confusion_matrix.png')
