import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

df = pd.read_csv('E:/car prediction/car_data.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nDtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDescriptive statistics:\n{df.describe(include='all')}")

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

df_clean = df.copy()

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col].fillna(median_val, inplace=True)
        print(f"Filled {df_clean[col].isnull().sum()} missing values in {col} with median: {median_val}")

categorical_cols = df_clean.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"Filled {df_clean[col].isnull().sum()} missing values in {col} with mode: {mode_val}")

print(f"\nMissing values after cleaning:\n{df_clean.isnull().sum()}")

Q1 = df_clean['price'].quantile(0.25)
Q3 = df_clean['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_price = df_clean[(df_clean['price'] < lower_bound) | (df_clean['price'] > upper_bound)]
print(f"\nPrice outliers (IQR method): {len(outliers_price)} samples")
print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")

Q1_km = df_clean['km_driven'].quantile(0.25)
Q3_km = df_clean['km_driven'].quantile(0.75)
IQR_km = Q3_km - Q1_km
lower_bound_km = Q1_km - 1.5 * IQR_km
upper_bound_km = Q3_km + 1.5 * IQR_km
outliers_km = df_clean[(df_clean['km_driven'] < lower_bound_km) | (df_clean['km_driven'] > upper_bound_km)]
print(f"KM driven outliers (IQR method): {len(outliers_km)} samples")
print(f"  Lower bound: {lower_bound_km:.2f}, Upper bound: {upper_bound_km:.2f}")

df_clean = df_clean[(df_clean['price'] >= lower_bound) & (df_clean['price'] <= upper_bound)]
df_clean = df_clean[(df_clean['km_driven'] >= lower_bound_km) & (df_clean['km_driven'] <= upper_bound_km)]
print(f"\nDataset shape after outlier removal: {df_clean.shape}")

z_scores = np.abs(stats.zscore(df_clean[['price', 'km_driven', 'year']]))
df_clean = df_clean[(z_scores < 3).all(axis=1)]
print(f"Dataset shape after Z-score outlier removal: {df_clean.shape}")

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

df_clean['car_age'] = 2024 - df_clean['year']
df_clean['km_per_year'] = df_clean['km_driven'] / df_clean['car_age'].replace(0, 1)
df_clean['price_per_km'] = df_clean['price'] / df_clean['km_driven'].replace(0, 1)

df_clean['is_luxury'] = df_clean['brand'].isin(['BMW', 'Mercedes', 'Audi', 'Lexus', 'Porsche', 'Tesla', 'Volvo']).astype(int)
df_clean['is_electric'] = (df_clean['fuel_type'] == 'Electric').astype(int)
df_clean['is_automatic'] = (df_clean['transmission'].isin(['Automatic', 'CVT', 'DSG'])).astype(int)

print(f"New features: car_age, km_per_year, price_per_km, is_luxury, is_electric, is_automatic")

categorical_features = ['brand', 'fuel_type', 'transmission', 'body_type', 'color']
numerical_features = ['year', 'km_driven', 'owner_count', 'has_accident', 'service_history',
                      'car_age', 'km_per_year', 'price_per_km', 'is_luxury', 'is_electric', 'is_automatic']

label_encoders = {}
df_encoded = df_clean.copy()
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col + '_label'] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le
    print(f"Label encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\n" + "=" * 60)
print("ONE-HOT ENCODING")
print("=" * 60)

df_ohe = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True, dtype=int)
print(f"Shape after one-hot encoding: {df_ohe.shape}")
print(f"New columns: {[c for c in df_ohe.columns if c not in df_clean.columns]}")

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

X = df_ohe.drop('price', axis=1)
y = df_ohe['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=42),
    'Lasso (alpha=0.1)': Lasso(alpha=0.1, random_state=42, max_iter=5000),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=15, min_samples_split=10),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20, 
                                            min_samples_split=5, n_jobs=-1)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name in ['Linear Regression', 'Ridge (alpha=1.0)', 'Lasso (alpha=0.1)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std()
    }
    
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
results_df = pd.DataFrame({
    'Model': [name for name in results.keys()],
    'MAE': [results[name]['MAE'] for name in results],
    'RMSE': [results[name]['RMSE'] for name in results],
    'R²': [results[name]['R2'] for name in results],
    'CV R² (mean)': [results[name]['CV_R2_mean'] for name in results],
    'CV R² (std)': [results[name]['CV_R2_std'] for name in results]
}).sort_values('R²', ascending=False)
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)

best_model_name = results_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
y_pred_best = results[best_model_name]['y_pred']
residuals = y_test - y_pred_best

print(f"Best model: {best_model_name}")
print(f"Residual mean: {residuals.mean():.2f}")
print(f"Residual std: {residuals.std():.2f}")
print(f"Residual skewness: {stats.skew(residuals):.4f}")

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(y_pred_best, residuals, alpha=0.5, s=10)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 2)
plt.scatter(y_test, y_pred_best, alpha=0.5, s=10)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
sorted_residuals = np.sort(residuals)
plt.plot(sorted_residuals, np.arange(len(sorted_residuals)) / len(sorted_residuals))
plt.xlabel('Residuals')
plt.ylabel('ECDF')
plt.title('Residual ECDF')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
if best_model_name in ['Random Forest', 'Decision Tree']:
    importances = best_model.feature_importances_
    feature_names = X.columns
    idx = np.argsort(importances)[-15:]
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 15 Features - {best_model_name}')
else:
    coef = np.abs(best_model.coef_)
    feature_names = X.columns
    idx = np.argsort(coef)[-15:]
    plt.barh(range(len(idx)), coef[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel('|Coefficient|')
    plt.title(f'Top 15 Features - {best_model_name}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:/car prediction/residual_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE (PERMUTATION)")
print("=" * 60)

perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
feature_importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Top 20 features by permutation importance:")
print(feature_importance_df.head(20).to_string(index=False))

plt.figure(figsize=(10, 8))
top_20 = feature_importance_df.head(20)
plt.barh(range(len(top_20)), top_20['importance_mean'], xerr=top_20['importance_std'], alpha=0.7)
plt.yticks(range(len(top_20)), top_20['feature'])
plt.xlabel('Permutation Importance')
plt.title(f'Top 20 Features - Permutation Importance ({best_model_name})')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('E:/car prediction/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

numeric_df = df_clean.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
price_corr = corr_matrix['price'].sort_values(ascending=False)
print("Correlation with price:")
print(price_corr)

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('E:/car prediction/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Best Model: {best_model_name}")
print(f"Test R²: {results[best_model_name]['R2']:.4f}")
print(f"Test MAE: {results[best_model_name]['MAE']:.2f}")
print(f"Test RMSE: {results[best_model_name]['RMSE']:.2f}")
print(f"\nTop 5 Price Drivers (Permutation Importance):")
for i, row in feature_importance_df.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance_mean']:.4f} (+/- {row['importance_std']:.4f})")

results_df.to_csv('E:/car prediction/model_comparison.csv', index=False)
feature_importance_df.to_csv('E:/car prediction/feature_importance.csv', index=False)
print("\nResults saved to CSV files.")