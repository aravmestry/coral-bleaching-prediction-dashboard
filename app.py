import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Coral Bleaching Prediction Dashboard",
    page_icon="🪸",
    layout="wide"
)

DATA_PATH = "data/global_bleaching_environmental.csv"

PREDICTORS = [
    "Latitude_Degrees", "Longitude_Degrees", "Depth_m",
    "SSTA", "SSTA_Mean", "SSTA_Maximum",
    "SSTA_DHW", "SSTA_DHWMean",
    "TSA", "TSA_Mean",
    "Turbidity", "Cyclone_Frequency"
]

TARGET = "Percent_Bleaching"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    data = df[PREDICTORS + [TARGET]].copy()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna()
    return data


@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


data = load_data()
X = data[PREDICTORS]
y = data[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = train_model(X_train, y_train)
pred = model.predict(X_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

st.title("🪸 Global Coral Bleaching Prediction Dashboard")

st.markdown(
    """
    This project uses a Random Forest model to predict **coral bleaching severity**
    from environmental conditions such as sea surface temperature anomalies,
    Degree Heating Weeks, turbidity, cyclone frequency, depth, latitude, and longitude.
    """
)

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Clean Observations", f"{len(data):,}")
m2.metric("Model", "Random Forest")
m3.metric("R² Score", f"{r2:.3f}")
m4.metric("RMSE", f"{rmse:.2f}")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🌍 Overview", "🔮 Prediction Tool", "📊 Model Results", "📁 Data Explorer"]
)

with tab1:
    st.subheader("Global Reef Observation Map")

    map_df = data.rename(columns={
        "Latitude_Degrees": "lat",
        "Longitude_Degrees": "lon"
    })

    sample_map = map_df.sample(min(6000, len(map_df)), random_state=42)

    fig_map = px.scatter_mapbox(
        sample_map,
        lat="lat",
        lon="lon",
        color=TARGET,
        size=TARGET,
        color_continuous_scale="Turbo",
        zoom=1,
        height=550,
        title="Global Coral Bleaching Observations",
        hover_data=[TARGET, "Depth_m", "SSTA_DHW"]
    )

    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r":0, "t":40, "l":0, "b":0})
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Bleaching Severity Distribution")

    fig_hist = px.histogram(
        data,
        x=TARGET,
        nbins=40,
        title="Distribution of Percent Bleaching",
        labels={TARGET: "Percent Bleaching"}
    )

    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Predict Coral Bleaching Severity")

    st.write("Adjust the environmental variables below to estimate bleaching severity.")

    user_inputs = {}
    col1, col2 = st.columns(2)

    for i, feature in enumerate(PREDICTORS):
        active_col = col1 if i % 2 == 0 else col2

        min_val = float(data[feature].min())
        max_val = float(data[feature].max())
        mean_val = float(data[feature].mean())

        if min_val == max_val:
            active_col.write(f"**{feature}:** {min_val}")
            user_inputs[feature] = min_val
        else:
            user_inputs[feature] = active_col.slider(
                feature,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    prediction = max(0, min(100, prediction))

    st.divider()

    st.metric("Predicted Percent Bleaching", f"{prediction:.2f}%")

    if prediction < 10:
        risk = "Low"
    elif prediction < 30:
        risk = "Moderate"
    elif prediction < 60:
        risk = "High"
    else:
        risk = "Severe"

    st.subheader(f"Risk Category: {risk}")

with tab3:
    st.subheader("Observed vs Predicted")

    results_df = pd.DataFrame({
        "Observed Percent Bleaching": y_test,
        "Predicted Percent Bleaching": pred
    })

    fig_scatter = px.scatter(
        results_df,
        x="Observed Percent Bleaching",
        y="Predicted Percent Bleaching",
        opacity=0.35,
        title="Observed vs Predicted Coral Bleaching"
    )

    fig_scatter.add_shape(
        type="line",
        x0=0, y0=0,
        x1=100, y1=100,
        line=dict(dash="dash")
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Variable": PREDICTORS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        importance_df,
        x="Importance",
        y="Variable",
        orientation="h",
        title="Environmental Drivers of Coral Bleaching"
    )

    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Model Performance")

    performance = pd.DataFrame({
        "Metric": ["R²", "MAE", "RMSE"],
        "Value": [r2, mae, rmse]
    })

    st.dataframe(performance, use_container_width=True)

with tab4:
    st.subheader("Cleaned Modeling Dataset")

    st.dataframe(data.head(1000), use_container_width=True)

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_coral_bleaching_model_data.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "Project 1: Predicting Coral Bleaching Severity from Environmental Conditions | "
    "Climate & Ecology Modeling Portfolio"
)