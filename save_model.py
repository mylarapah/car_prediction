import pickle
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

df = pd.read_csv('E:/car prediction/car_data.csv')

df_clean = df.copy()

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)

categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)

Q1 = df_clean['price'].quantile(0.25)
Q3 = df_clean['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

Q1_km = df_clean['km_driven'].quantile(0.25)
Q3_km = df_clean['km_driven'].quantile(0.75)
IQR_km = Q3_km - Q1_km
lower_bound_km = Q1_km - 1.5 * IQR_km
upper_bound_km = Q3_km + 1.5 * IQR_km

df_clean = df_clean[(df_clean['price'] >= lower_bound) & (df_clean['price'] <= upper_bound)]
df_clean = df_clean[(df_clean['km_driven'] >= lower_bound_km) & (df_clean['km_driven'] <= upper_bound_km)]

z_scores = np.abs(stats.zscore(df_clean[['price', 'km_driven', 'year']]))
df_clean = df_clean[(z_scores < 3).all(axis=1)]

df_clean['car_age'] = 2024 - df_clean['year']
df_clean['km_per_year'] = df_clean['km_driven'] / df_clean['car_age'].replace(0, 1)
df_clean['price_per_km'] = df_clean['price'] / df_clean['km_driven'].replace(0, 1)

df_clean['is_luxury'] = df_clean['brand'].isin(['BMW', 'Mercedes', 'Audi', 'Lexus', 'Porsche', 'Tesla', 'Volvo']).astype(int)
df_clean['is_electric'] = (df_clean['fuel_type'] == 'Electric').astype(int)
df_clean['is_automatic'] = (df_clean['transmission'].isin(['Automatic', 'CVT', 'DSG'])).astype(int)

categorical_features = ['brand', 'fuel_type', 'transmission', 'body_type', 'color']

label_encoders = {}
df_encoded = df_clean.copy()
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_label'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

df_ohe = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True, dtype=int)

X = df_ohe.drop('price', axis=1)
y = df_ohe['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20, 
                                    min_samples_split=5, n_jobs=-1)
best_model.fit(X_train, y_train)

feature_columns = X.columns.tolist()

with open('E:/car prediction/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('E:/car prediction/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('E:/car prediction/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

with open('E:/car prediction/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("Model and artifacts saved successfully!")
print(f"Feature columns: {len(feature_columns)}")
print(f"Label encoders: {list(label_encoders.keys())}")