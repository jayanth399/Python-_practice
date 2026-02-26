import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="✈️ Flight Price Predictor", layout="wide")

st.title("✈️ Flight Price Prediction – ML Pipeline")
st.markdown("Upload your `Data_Train.xlsx` file to run the full pipeline.")

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Data_Train.xlsx", type=["xlsx"])

if uploaded_file is None:
    st.info("Please upload the dataset to get started.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD & DISPLAY RAW DATA
# ─────────────────────────────────────────────
df = pd.read_excel(uploaded_file)

with st.expander("📋 Raw Data Preview"):
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    buf = []
    df.info(buf=buf)
    # show null counts neatly
    null_counts = df.isnull().sum()
    st.write("**Null values per column:**")
    st.dataframe(null_counts[null_counts >= 0].rename("Nulls").to_frame(), use_container_width=True)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
st.subheader("🔧 Feature Engineering")

with st.spinner("Processing features…"):
    # Date features
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Day"]   = df["Date_of_Journey"].dt.day
    df["Month"] = df["Date_of_Journey"].dt.month
    df["Year"]  = df["Date_of_Journey"].dt.year
    df.drop("Date_of_Journey", axis=1, inplace=True)

    # Departure hour / minute
    df["Dep_hour"] = df["Dep_Time"].str.split(":").str[0].astype(int)
    df["Dep_min"]  = df["Dep_Time"].str.split(":").str[1].astype(int)
    df.drop("Dep_Time", axis=1, inplace=True)

    # Arrival hour / minute
    df["Arr_hour"] = df["Arrival_Time"].str.split(" ").str[0].str.split(":").str[0].astype(int)
    df["Arr_min"]  = df["Arrival_Time"].str.split(" ").str[0].str.split(":").str[1].astype(int)
    df.drop("Arrival_Time", axis=1, inplace=True)

    # Total stops → numeric
    df["Total_Stops"] = df["Total_Stops"].replace("non-stop", "0 stop")
    df["Total_Stops"] = df["Total_Stops"].astype(str).str.extract(r"(\d+)")[0].fillna(0).astype(int)

    # Route → split columns
    for i, col in enumerate(["route1", "route2", "route3", "route4", "route5"], start=1):
        df[col] = df["Route"].str.split("→").str[i - 1].str.strip() if i == 1 else \
                  df["Route"].str.split("→").str[i - 1].str.strip()
    # redo cleanly
    route_parts = df["Route"].str.split("→")
    for i in range(5):
        df[f"route{i+1}"] = route_parts.str[i].str.strip() if route_parts.str[i].notna().any() else np.nan
    df.drop("Route", axis=1, inplace=True)

    # Label-encode all object columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Log-transform Price
    df["Price"] = np.log(df["Price"])

st.success("Feature engineering complete!")
with st.expander("📐 Processed Data Preview"):
    st.dataframe(df.head(), use_container_width=True)

# ─────────────────────────────────────────────
# EDA CHARTS
# ─────────────────────────────────────────────
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Distribution of Flight Price (log-transformed)**")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(df["Price"], bins=30, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Log Price")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Price Distribution")
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.markdown("**Model Comparison – R² Score**")
    # placeholder until models are trained; filled below

# ─────────────────────────────────────────────
# TRAIN / EVALUATE MODELS
# ─────────────────────────────────────────────
st.subheader("🤖 Model Training & Evaluation")

X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression":        LinearRegression(),
    "Decision Tree Regressor":  DecisionTreeRegressor(random_state=42),
    "KNN Regressor":            KNeighborsRegressor(n_neighbors=5),
    "Random Forest Regressor":  RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR":                      SVR(kernel="rbf"),
}

results = []
predictions = {}

progress = st.progress(0, text="Training models…")
for idx, (name, mdl) in enumerate(models.items()):
    mdl.fit(X_train, y_train)
    y_pred = mdl.predict(X_test)
    predictions[name] = y_pred
    results.append({
        "Model":    name,
        "MAE":      mean_absolute_error(y_test, y_pred),
        "MSE":      mean_squared_error(y_test, y_pred),
        "R² Score": r2_score(y_test, y_pred),
    })
    progress.progress((idx + 1) / len(models), text=f"Trained: {name}")

progress.empty()

results_df = pd.DataFrame(results).sort_values("R² Score", ascending=False).reset_index(drop=True)

st.markdown("### 📈 Model Comparison")
st.dataframe(results_df.style.format({"MAE": "{:.4f}", "MSE": "{:.4f}", "R² Score": "{:.4f}"}),
             use_container_width=True)

# Bar chart
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(results_df))]
    ax2.barh(results_df["Model"][::-1], results_df["R² Score"][::-1], color=colors[::-1], edgecolor="black")
    ax2.set_xlabel("R² Score")
    ax2.set_title("R² Score Comparison")
    ax2.set_xlim(0, 1)
    st.pyplot(fig2)
    plt.close(fig2)

# ─────────────────────────────────────────────
# BEST MODEL DETAILS
# ─────────────────────────────────────────────
best_name = results_df.iloc[0]["Model"]
st.success(f"🏆 Best Model: **{best_name}** with R² = {results_df.iloc[0]['R² Score']:.4f}")

st.markdown(f"### 🔍 Actual vs Predicted – {best_name}")
best_pred = predictions[best_name]
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.scatter(y_test, best_pred, alpha=0.4, color="royalblue", edgecolors="none", s=15)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax3.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax3.set_xlabel("Actual (log price)")
ax3.set_ylabel("Predicted (log price)")
ax3.set_title(f"Actual vs Predicted – {best_name}")
ax3.legend()
st.pyplot(fig3)
plt.close(fig3)

st.markdown("---")
st.caption("Pipeline mirrors the original Jupyter notebook (ml.ipynb) | Built with Streamlit")