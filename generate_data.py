import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 5000

brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Audi', 'Volkswagen', 'Hyundai', 'Kia', 'Nissan',
          'Mazda', 'Subaru', 'Lexus', 'Chevrolet', 'Jeep', 'Volvo', 'Porsche', 'Tesla', 'Mitsubishi', 'Suzuki']
fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'CNG']
transmissions = ['Manual', 'Automatic', 'CVT', 'DSG']
body_types = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Convertible', 'Wagon', 'Pickup']
colors = ['White', 'Black', 'Silver', 'Gray', 'Blue', 'Red', 'Brown', 'Green', 'Yellow', 'Orange']

brand_base_price = {
    'Toyota': 25000, 'Honda': 24000, 'Ford': 22000, 'BMW': 45000, 'Mercedes': 50000,
    'Audi': 42000, 'Volkswagen': 28000, 'Hyundai': 20000, 'Kia': 19000, 'Nissan': 21000,
    'Mazda': 23000, 'Subaru': 26000, 'Lexus': 48000, 'Chevrolet': 24000, 'Jeep': 30000,
    'Volvo': 38000, 'Porsche': 80000, 'Tesla': 55000, 'Mitsubishi': 18000, 'Suzuki': 16000
}

fuel_multiplier = {'Petrol': 1.0, 'Diesel': 1.15, 'Hybrid': 1.25, 'Electric': 1.4, 'CNG': 0.9}
transmission_multiplier = {'Manual': 0.95, 'Automatic': 1.05, 'CVT': 1.02, 'DSG': 1.08}
body_multiplier = {'Sedan': 1.0, 'SUV': 1.2, 'Hatchback': 0.85, 'Coupe': 1.15, 'Convertible': 1.25, 'Wagon': 1.05, 'Pickup': 1.1}

data = []

for _ in range(n_samples):
    brand_probs = np.array([0.12, 0.11, 0.1, 0.07, 0.06, 0.06, 0.07, 0.08, 0.07, 0.06,
                              0.04, 0.03, 0.04, 0.04, 0.03, 0.02, 0.01, 0.02, 0.02, 0.02])
    brand_probs = brand_probs / brand_probs.sum()
    brand = np.random.choice(brands, p=brand_probs)
    year = np.random.randint(2010, 2024)
    age = 2024 - year
    
    base_price = brand_base_price[brand]
    base_price *= fuel_multiplier[np.random.choice(fuel_types, p=[0.45, 0.25, 0.15, 0.1, 0.05])]
    base_price *= transmission_multiplier[np.random.choice(transmissions, p=[0.35, 0.4, 0.15, 0.1])]
    base_price *= body_multiplier[np.random.choice(body_types, p=[0.25, 0.3, 0.2, 0.08, 0.05, 0.07, 0.05])]
    
    km_driven = np.random.lognormal(mean=10.5, sigma=0.8) * (age / 5)
    km_driven = np.clip(km_driven, 1000, 300000)
    
    depreciation = 0.85 ** age
    mileage_factor = np.exp(-km_driven / 150000)
    price = base_price * depreciation * mileage_factor
    
    noise = np.random.normal(1, 0.08)
    price *= noise
    
    if np.random.random() < 0.02:
        price *= np.random.uniform(0.3, 0.6)
    
    fuel_type = np.random.choice(fuel_types, p=[0.45, 0.25, 0.15, 0.1, 0.05])
    transmission = np.random.choice(transmissions, p=[0.35, 0.4, 0.15, 0.1])
    body_type = np.random.choice(body_types, p=[0.25, 0.3, 0.2, 0.08, 0.05, 0.07, 0.05])
    color = np.random.choice(colors)
    
    owner_count = np.random.choice([1, 2, 3, 4], p=[0.55, 0.3, 0.1, 0.05])
    if owner_count > 1:
        price *= (0.95 ** (owner_count - 1))
    
    has_accident = np.random.choice([0, 1], p=[0.85, 0.15])
    if has_accident:
        price *= np.random.uniform(0.7, 0.9)
    
    service_history = np.random.choice([0, 1], p=[0.3, 0.7])
    if service_history:
        price *= 1.05
    
    data.append({
        'brand': brand,
        'year': year,
        'km_driven': round(km_driven),
        'fuel_type': fuel_type,
        'transmission': transmission,
        'body_type': body_type,
        'color': color,
        'owner_count': owner_count,
        'has_accident': has_accident,
        'service_history': service_history,
        'price': round(price, 2)
    })

df = pd.DataFrame(data)

missing_indices = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
df.loc[missing_indices[:int(0.015 * n_samples)], 'km_driven'] = np.nan
df.loc[missing_indices[int(0.015 * n_samples):], 'service_history'] = np.nan

outlier_indices = np.random.choice(df.index, size=int(0.01 * n_samples), replace=False)
df.loc[outlier_indices, 'price'] *= np.random.uniform(3, 5, size=len(outlier_indices))
df.loc[outlier_indices, 'km_driven'] = np.random.uniform(500, 2000, size=len(outlier_indices))

df.to_csv('E:/car prediction/car_data.csv', index=False)
print(f"Generated {len(df)} samples")
print(df.head())
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nPrice stats:\n{df['price'].describe()}")