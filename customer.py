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