import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv('E:/car prediction/car_data.csv')
    return df

@st.cache_resource
def load_model():
    import pickle
    with open('E:/car prediction/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('E:/car prediction/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('E:/car prediction/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('E:/car prediction/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, scaler, feature_columns, label_encoders

def prepare_input_data(input_dict, feature_columns, label_encoders, scaler):
    df = pd.DataFrame([input_dict])
    
    df['car_age'] = 2024 - df['year']
    df['km_per_year'] = df['km_driven'] / df['car_age'].replace(0, 1)
    df['price_per_km'] = 1.0
    
    df['is_luxury'] = df['brand'].isin(['BMW', 'Mercedes', 'Audi', 'Lexus', 'Porsche', 'Tesla', 'Volvo']).astype(int)
    df['is_electric'] = (df['fuel_type'] == 'Electric').astype(int)
    df['is_automatic'] = (df['transmission'].isin(['Automatic', 'CVT', 'DSG'])).astype(int)
    
    categorical_features = ['brand', 'fuel_type', 'transmission', 'body_type', 'color']
    for col in categorical_features:
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
    
    df_ohe = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=int)
    
    for col in feature_columns:
        if col not in df_ohe.columns:
            df_ohe[col] = 0
    
    df_ohe = df_ohe[feature_columns]
    
    return scaler.transform(df_ohe), df_ohe

def main():
    st.title("🚗 Used Car Price Prediction")
    st.markdown("### Predict resale value using Machine Learning")
    
    df = load_data()
    
    model, scaler, feature_columns, label_encoders = load_model()
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Price", "📊 Data Explorer", "📈 Model Performance", "📋 About"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Vehicle Details")
            
            brand = st.selectbox("Brand", sorted(df['brand'].unique()))
            year = st.slider("Manufacturing Year", 2010, 2024, 2020)
            km_driven = st.number_input("Kilometres Driven", min_value=0, max_value=300000, value=50000, step=1000)
            fuel_type = st.selectbox("Fuel Type", sorted(df['fuel_type'].unique()))
            transmission = st.selectbox("Transmission", sorted(df['transmission'].unique()))
            
        with col2:
            st.subheader("Additional Details")
            
            body_type = st.selectbox("Body Type", sorted(df['body_type'].unique()))
            color = st.selectbox("Color", sorted(df['color'].unique()))
            owner_count = st.selectbox("Number of Previous Owners", [1, 2, 3, 4])
            has_accident = st.selectbox("Accident History", ["No", "Yes"])
            service_history = st.selectbox("Service History Available", ["Yes", "No"])
        
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            input_data = {
                'brand': brand,
                'year': year,
                'km_driven': km_driven,
                'fuel_type': fuel_type,
                'transmission': transmission,
                'body_type': body_type,
                'color': color,
                'owner_count': owner_count,
                'has_accident': 1 if has_accident == "Yes" else 0,
                'service_history': 1 if service_history == "Yes" else 0
            }
            
            X_scaled, X_raw = prepare_input_data(input_data, feature_columns, label_encoders, scaler)
            prediction = model.predict(X_scaled)[0]
            
            st.success(f"### Estimated Price: ₹{prediction:,.2f}")
            
            col_a, col_b, col_c = st.columns(3)
            car_age = 2024 - year
            with col_a:
                st.metric("Car Age", f"{car_age} years")
            with col_b:
                st.metric("Km/Year", f"{km_driven/car_age if car_age > 0 else 0:,.0f}")
            with col_c:
                brand_avg = df[df['brand'] == brand]['price'].mean()
                st.metric("Brand Avg Price", f"₹{brand_avg:,.0f}")
            
            similar_cars = df[
                (df['brand'] == brand) & 
                (df['year'].between(year-2, year+2)) &
                (df['km_driven'].between(km_driven*0.7, km_driven*1.3))
            ]
            if len(similar_cars) > 0:
                st.info(f"📊 Found {len(similar_cars)} similar cars in database. Avg price: ₹{similar_cars['price'].mean():,.0f}")
    
    with tab2:
        st.subheader("Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Avg Price", f"₹{df['price'].mean():,.0f}")
        with col3:
            st.metric("Price Range", f"₹{df['price'].min():,.0f} - ₹{df['price'].max():,.0f}")
        with col4:
            st.metric("Brands", df['brand'].nunique())
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x='price', nbins=50, title='Price Distribution', 
                              color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='brand', y='price', title='Price by Brand')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='year', y='price', color='fuel_type', 
                           title='Price vs Year by Fuel Type', opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='km_driven', y='price', color='transmission',
                           title='Price vs Km Driven by Transmission', opacity=0.6)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Model Performance Comparison")
        
        results_df = pd.read_csv('E:/car prediction/model_comparison.csv')
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=('MAE', 'RMSE', 'R² Score'))
        
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['MAE'], name='MAE', marker_color='lightblue'), row=1, col=1)
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['RMSE'], name='RMSE', marker_color='lightcoral'), row=1, col=2)
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df['R²'], name='R²', marker_color='lightgreen'), row=1, col=3)
        
        fig.update_layout(height=400, showlegend=False, title_text="Model Metrics Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(results_df.style.format({
            'MAE': '{:.2f}',
            'RMSE': '{:.2f}',
            'R²': '{:.4f}',
            'CV R² (mean)': '{:.4f}',
            'CV R² (std)': '{:.4f}'
        }).background_gradient(subset=['R²'], cmap='RdYlGn'), use_container_width=True)
        
        st.subheader("Top Features Driving Price")
        feat_df = pd.read_csv('E:/car prediction/feature_importance.csv').head(15)
        
        fig = px.bar(feat_df.sort_values('importance_mean', ascending=True), 
                     x='importance_mean', y='feature', orientation='h',
                     title='Permutation Feature Importance (Top 15)',
                     error_x='importance_std')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Residual Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.image('E:/car prediction/residual_analysis.png', caption='Residual Analysis Plots')
        with col2:
            st.image('E:/car prediction/feature_importance.png', caption='Feature Importance')
    
    with tab4:
        st.subheader("About This Project")
        st.markdown("""
        ### Car Price Prediction System
        
        This application predicts the resale price of used cars using machine learning models trained on 
        synthetic data representing realistic market conditions.
        
        **Features Used:**
        - Brand, Manufacturing Year, Kilometres Driven
        - Fuel Type, Transmission, Body Type, Color
        - Owner Count, Accident History, Service History
        
        **Engineered Features:**
        - Car Age, Km per Year, Price per Km
        - Luxury Brand Flag, Electric Vehicle Flag, Automatic Transmission Flag
        
        **Models Compared:**
        1. **Random Forest** (Best) - R² = 0.981
        2. Decision Tree - R² = 0.949
        3. Lasso Regression - R² = 0.805
        4. Ridge Regression - R² = 0.805
        5. Linear Regression - R² = 0.805
        
        **Key Price Drivers:**
        - Price per Km (engineered feature)
        - Kilometres Driven
        - Km per Year
        - Luxury Brand Status
        - Car Age
        
        **Tech Stack:**
        - Python, scikit-learn, pandas, numpy
        - Streamlit for frontend
        - Plotly for visualizations
        """)

if __name__ == "__main__":
    main()