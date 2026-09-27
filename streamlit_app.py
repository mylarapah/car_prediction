"""Used Car Price Prediction — Streamlit app (deployment-ready).

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy (Streamlit Community Cloud):
    1. Push this folder to GitHub (include streamlit_app.py, requirements.txt,
       best_model.pkl, scaler.pkl, feature_columns.pkl, label_encoders.pkl,
       car_data.csv, model_comparison.csv, feature_importance.csv, *.png).
    2. On share.streamlit.io -> New app -> select repo -> main file:
       streamlit_app.py
"""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.pkl"
ENCODERS_PATH = BASE_DIR / "label_encoders.pkl"
DATA_PATH = BASE_DIR / "car_data.csv"
METRICS_PATH = BASE_DIR / "model_comparison.csv"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"

CATEGORICAL_FEATURES = ["brand", "fuel_type", "transmission", "body_type", "color"]
LUXURY_BRANDS = ["BMW", "Mercedes", "Audi", "Lexus", "Porsche", "Tesla", "Volvo"]
CURRENT_YEAR = 2024

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- loaders
@st.cache_resource
def load_artifacts():
    """Load model + preprocessing artifacts using paths relative to this file."""
    missing = [p.name for p in (MODEL_PATH, FEATURES_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifact(s): {missing}. "
            "Run `python save_model.py` first, or include the .pkl files in your deploy."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_columns = pickle.load(f)
    scaler = None
    if SCALER_PATH.exists():
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    label_encoders = {}
    if ENCODERS_PATH.exists():
        with open(ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
    return model, scaler, feature_columns, label_encoders


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


@st.cache_data
def median_price_per_km(_df: pd.DataFrame | None) -> float:
    """Median price/km from training data — better placeholder than a constant."""
    try:
        if _df is not None and {"price", "km_driven"}.issubset(_df.columns):
            s = _df["price"] / _df["km_driven"].replace(0, np.nan)
            m = float(s.median())
            if np.isfinite(m):
                return m
    except Exception:
        pass
    return 1.0

# ------------------------------------------------------- preprocessing
def build_features(input_dict: dict, feature_columns: list, ppk_fallback: float = 1.0) -> pd.DataFrame:
    """Replicate training feature engineering, then one-hot encode.

    Must match save_model.py:
      car_age, km_per_year, price_per_km, is_luxury, is_electric, is_automatic
      + pd.get_dummies(..., columns=CATEGORICAL_FEATURES, drop_first=True)
    """
    df = pd.DataFrame([input_dict])

    df["car_age"] = CURRENT_YEAR - df["year"]
    df["km_per_year"] = df["km_driven"] / df["car_age"].replace(0, 1)
    # Target-derived at train time; unknown at inference -> dataset median.
    df["price_per_km"] = ppk_fallback

    df["is_luxury"] = df["brand"].isin(LUXURY_BRANDS).astype(int)
    df["is_electric"] = (df["fuel_type"] == "Electric").astype(int)
    df["is_automatic"] = df["transmission"].isin(["Automatic", "CVT", "DSG"]).astype(int)

    # NOTE: do NOT label-encode before get_dummies — training one-hot encoded
    # the raw strings, so inference must too, then align columns exactly.
    df_ohe = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int)
    df_ohe = df_ohe.reindex(columns=feature_columns, fill_value=0)
    return df_ohe


def predict_price(model, scaler, X: pd.DataFrame) -> float:
    """Tree models were trained on unscaled data; linear models on scaled data."""
    model_name = type(model).__name__
    tree_models = {"RandomForestRegressor", "DecisionTreeRegressor",
                   "GradientBoostingRegressor", "ExtraTreesRegressor"}
    if model_name in tree_models or scaler is None:
        return float(model.predict(X)[0])
    return float(model.predict(scaler.transform(X))[0])

# ---------------------------------------------------------------- UI
def main():
    st.title("🚗 Used Car Price Prediction")
    st.markdown("### Predict resale value using Machine Learning")

    try:
        model, scaler, feature_columns, _label_encoders = load_artifacts()
    except Exception as e:
        st.error(f"⚠️ Could not load model artifacts. {e}")
        st.stop()

    df = load_data()
    if df is None:
        st.warning("`car_data.csv` not found — prediction still works, but explorer charts are disabled.")
    ppk_fallback = median_price_per_km(df)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔮 Predict Price", "📁 Batch Predict", "📊 Data Explorer", "📈 Model Performance"]
    )

    # ---- Tab 1: single prediction ----
    with tab1:
        col1, col2 = st.columns(2)
        if df is not None:
            brands = sorted(df["brand"].dropna().unique())
            fuels = sorted(df["fuel_type"].dropna().unique())
            transmissions = sorted(df["transmission"].dropna().unique())
            bodies = sorted(df["body_type"].dropna().unique())
            colors = sorted(df["color"].dropna().unique())
            year_min, year_max = int(df["year"].min()), int(df["year"].max())
        else:  # sensible defaults if CSV missing
            brands = ["Toyota", "Honda", "BMW", "Mercedes", "Audi", "Tesla"]
            fuels = ["Petrol", "Diesel", "Hybrid", "Electric", "CNG"]
            transmissions = ["Manual", "Automatic", "CVT", "DSG"]
            bodies = ["Sedan", "SUV", "Hatchback", "Coupe", "Convertible", "Wagon", "Pickup"]
            colors = ["Black", "White", "Silver", "Red", "Blue"]
            year_min, year_max = 2010, 2024

        with col1:
            st.subheader("Vehicle Details")
            brand = st.selectbox("Brand", brands)
            year = st.slider("Manufacturing Year", year_min, year_max, min(2020, year_max))
            km_driven = st.number_input("Kilometres Driven", min_value=0, max_value=300000, value=50000, step=1000)
            fuel_type = st.selectbox("Fuel Type", fuels)
            transmission = st.selectbox("Transmission", transmissions)
        with col2:
            st.subheader("Additional Details")
            body_type = st.selectbox("Body Type", bodies)
            color = st.selectbox("Color", colors)
            owner_count = st.selectbox("Number of Previous Owners", [1, 2, 3, 4])
            has_accident = st.selectbox("Accident History", ["No", "Yes"])
            service_history = st.selectbox("Service History Available", ["Yes", "No"])

        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            input_data = {
                "brand": brand, "year": year, "km_driven": km_driven,
                "fuel_type": fuel_type, "transmission": transmission,
                "body_type": body_type, "color": color,
                "owner_count": owner_count,
                "has_accident": 1 if has_accident == "Yes" else 0,
                "service_history": 1 if service_history == "Yes" else 0,
            }
            X = build_features(input_data, feature_columns, ppk_fallback)
            prediction = predict_price(model, scaler, X)
            st.success(f"### Estimated Price: ₹{prediction:,.2f}")

            car_age = CURRENT_YEAR - year
            c1, c2, c3 = st.columns(3)
            c1.metric("Car Age", f"{car_age} years")
            c2.metric("Km / Year", f"{km_driven / car_age if car_age > 0 else 0:,.0f}")
            if df is not None and "price" in df.columns:
                brand_avg = df.loc[df["brand"] == brand, "price"].mean()
                c3.metric("Brand Avg Price", f"₹{brand_avg:,.0f}")
                similar = df[
                    (df["brand"] == brand)
                    & (df["year"].between(year - 2, year + 2))
                    & (df["km_driven"].between(km_driven * 0.7, km_driven * 1.3))
                ]
                if len(similar):
                    st.info(f"📊 {len(similar)} similar cars in database. Avg price: ₹{similar['price'].mean():,.0f}")

    # ---- Tab 2: batch prediction ----
    with tab2:
        st.subheader("Batch prediction from CSV")
        st.markdown(
            "Upload a CSV with columns: `brand, year, km_driven, fuel_type, transmission,"
            " body_type, color, owner_count, has_accident, service_history` "
            "(`has_accident` / `service_history` as 0/1 or Yes/No)."
        )
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                bdf = pd.read_csv(uploaded)
                for col in ("has_accident", "service_history"):
                    if col in bdf.columns and bdf[col].dtype == object:
                        bdf[col] = bdf[col].map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).fillna(bdf[col])
                        bdf[col] = pd.to_numeric(bdf[col], errors="coerce").fillna(0).astype(int)
                preds = []
                for _, row in bdf.iterrows():
                    X = build_features(row.to_dict(), feature_columns, ppk_fallback)
                    preds.append(predict_price(model, scaler, X))
                bdf["predicted_price"] = preds
                st.dataframe(bdf, use_container_width=True)
                st.download_button(
                    "⬇️ Download predictions",
                    bdf.to_csv(index=False).encode("utf-8"),
                    file_name="car_price_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

    # ---- Tab 3: data explorer ----
    with tab3:
        st.subheader("Dataset Overview")
        if df is None:
            st.info("Add `car_data.csv` next to this file to enable the explorer.")
        else:
            import plotly.express as px

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Samples", f"{len(df):,}")
            m2.metric("Avg Price", f"₹{df['price'].mean():,.0f}")
            m3.metric("Price Range", f"₹{df['price'].min():,.0f} – ₹{df['price'].max():,.0f}")
            m4.metric("Brands", df["brand"].nunique())

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.histogram(df, x="price", nbins=50, title="Price Distribution"),
                    use_container_width=True,
                )
            with c2:
                fig = px.box(df, x="brand", y="price", title="Price by Brand")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    px.scatter(df, x="year", y="price", color="fuel_type",
                               title="Price vs Year by Fuel Type", opacity=0.6),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    px.scatter(df, x="km_driven", y="price", color="transmission",
                               title="Price vs Km Driven by Transmission", opacity=0.6),
                    use_container_width=True,
                )

    # ---- Tab 4: model performance ----
    with tab4:
        st.subheader("Model Performance Comparison")
        if METRICS_PATH.exists():
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            results_df = pd.read_csv(METRICS_PATH)
            fig = make_subplots(rows=1, cols=3, subplot_titles=("MAE", "RMSE", "R² Score"))
            fig.add_trace(go.Bar(x=results_df["Model"], y=results_df["MAE"], name="MAE"), row=1, col=1)
            fig.add_trace(go.Bar(x=results_df["Model"], y=results_df["RMSE"], name="RMSE"), row=1, col=2)
            fig.add_trace(go.Bar(x=results_df["Model"], y=results_df["R²"], name="R²"), row=1, col=3)
            fig.update_layout(height=400, showlegend=False, title_text="Model Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info("`model_comparison.csv` not found.")
        if IMPORTANCE_PATH.exists():
            import plotly.express as px

            feat_df = pd.read_csv(IMPORTANCE_PATH).head(15)
            fig = px.bar(feat_df.sort_values("importance_mean"),
                         x="importance_mean", y="feature", orientation="h",
                         title="Top 15 Features (Permutation Importance)",
                         error_x="importance_std" if "importance_std" in feat_df else None)
            st.plotly_chart(fig, use_container_width=True)
        for img in ("residual_analysis.png", "feature_importance.png"):
            p = BASE_DIR / img
            if p.exists():
                st.image(str(p), caption=img)

    st.divider()
    st.caption("Run locally with `streamlit run streamlit_app.py` · Artifacts loaded relative to this file.")


if __name__ == "__main__":
    main()
