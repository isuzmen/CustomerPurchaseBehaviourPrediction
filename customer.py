import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    RocCurveDisplay, ConfusionMatrixDisplay
)

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

df = pd.read_csv('marketing_campaign.csv', sep='\t')

df['Income'] = df['Income'].fillna(df['Income'].median())

Q95 = df['Income'].quantile(0.95)
df['Income'] = np.where(df['Income'] > Q95, Q95, df['Income'])

df['Age'] = 2023 - df['Year_Birth']
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Membership_Days'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days

df['Marital_Status'] = df['Marital_Status'].replace({'Alone': 'Single', 'Absurd': 'Single', 'YOLO': 'Single'})
df = pd.get_dummies(df, columns=['Education', 'Marital_Status'], drop_first=True)

df.drop(columns=['Z_CostContact', 'Z_Revenue', 'ID', 'Year_Birth', 'Dt_Customer', 'Kidhome', 'Teenhome'], inplace=True)

corr_matrix = df.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Korelasyon Matrisi Isı Haritası")
plt.tight_layout()
plt.show()

high_corr = [col for col in corr_matrix.columns if any(abs(corr_matrix[col]) > 0.85) and col != 'Response']

print("\n Yüksek Korelasyonlu Özellikler (|r| > 0.85):", high_corr)

print("\n Response ile En Yüksek Korelasyonlar:")
print(corr_matrix['Response'].sort_values(ascending=False).head(10))

print("\n Response ile En Düşük Korelasyonlar:")
print(corr_matrix['Response'].sort_values().head(10))

print(f"\n Veri Seti Boyutu: {df.shape}")

X = df.drop('Response', axis=1)
y = df['Response']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(f"\n Eğitim Seti (orijinal): {X_train.shape}, Sınıf Dağılımı: {dict(pd.Series(y_train).value_counts())}")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f" Eğitim Seti (SMOTE sonrası): {X_train_res.shape}, Sınıf Dağılımı: {dict(pd.Series(y_train_res).value_counts())}")
print(f" Test Seti: {X_test.shape}, Sınıf Dağılımı: {dict(pd.Series(y_test).value_counts())}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

models = {
    "Lojistik Regresyon": {
        "model": LogisticRegression(max_iter=1000),
        "params": {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    },
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {'n_estimators': [100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
    },
    "SVM": {
        "model": SVC(probability=True),
        "params": {'C': [0.1, 1, 10], 'gamma': ['scale'], 'kernel': ['rbf', 'linear']}
    },
    "kNN": {
        "model": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(),
        "params": {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {}
    }
}

results = []

plt.figure(figsize=(18, 5))
ax1 = plt.subplot(131)

for name, config in models.items():
    grid = GridSearchCV(config["model"], config["params"], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_scaled, y_train_res)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else np.zeros_like(y_pred)

    cv_f1 = cross_val_score(best_model, X_train_scaled, y_train_res, cv=5, scoring='f1').mean()
    cv_auc = cross_val_score(best_model, X_train_scaled, y_train_res, cv=5, scoring='roc_auc').mean()
    cv_prc = cross_val_score(best_model, X_train_scaled, y_train_res, cv=5, scoring='average_precision').mean()

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1": round(f1_score(y_test, y_pred), 3),
        "AUC-ROC": round(roc_auc_score(y_test, y_proba), 3) if y_proba.any() else None,
        "PR AUC": round(average_precision_score(y_test, y_proba), 3) if y_proba.any() else None,
        "MCC": round(matthews_corrcoef(y_test, y_pred), 3),
        "CV F1": round(cv_f1, 3),
        "CV AUC": round(cv_auc, 3),
        "CV PRC": round(cv_prc, 3)
    })

    if y_proba.any():
        RocCurveDisplay.from_estimator(best_model, X_test_scaled, y_test, ax=ax1, name=name)

ax1.set_title("ROC Eğrileri")

results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False)
print("\n Model Performansları:\n", results_df)

best_model_name = results_df.iloc[0]["Model"]
print(f"\n En İyi Model: {best_model_name}")

if best_model_name == "Random Forest":
    best_model = GridSearchCV(models[best_model_name]["model"], models[best_model_name]["params"], cv=5, scoring='f1', n_jobs=-1)
    best_model.fit(X_train_scaled, y_train_res)
    best_model = best_model.best_estimator_

    plt.figure(figsize=(6, 4))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test_scaled, y_test)
    plt.title(f"{best_model_name} Confusion Matrix")

    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
    print("\n En Önemli 15 Özellik:\n", feat_imp)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index)
    plt.title(f"{best_model_name} Önemli Özellikler")
    plt.tight_layout()

plt.show()