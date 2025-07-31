import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

st.set_page_config(page_title="Universal Energy Analytics", layout="wide")
st.title("🔌 Universal Energy Consumption Dashboard")

uploaded_file = st.file_uploader("📁 Upload your energy dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📋 Column Overview")
    st.write(df.columns.tolist())

    # Detect or select datetime column
    datetime_guess = next((col for col in df.columns if "date" in col.lower() or "time" in col.lower()), None)
    datetime_col = st.selectbox("🕒 Select the Date/Time column", df.columns, index=df.columns.get_loc(datetime_guess) if datetime_guess else 0)

    # Parse and set datetime index
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df.dropna(subset=[datetime_col], inplace=True)
    df.set_index(datetime_col, inplace=True)

    # Detect or select energy column
    energy_guess = next((col for col in df.columns if "energy" in col.lower() or "appliance" in col.lower() or "usage" in col.lower()), None)
    energy_col = st.selectbox("⚡ Select the Energy Usage column", df.columns, index=df.columns.get_loc(energy_guess) if energy_guess else 0)

    # Convert energy column to numeric
    df[energy_col] = pd.to_numeric(df[energy_col], errors='coerce')
    df.dropna(subset=[energy_col], inplace=True)

    st.subheader("🔍 Preview of Data")
    st.dataframe(df.head())

    # Plot energy usage
    st.subheader("📈 Energy Usage Over Time")
    st.line_chart(df[energy_col])

    st.subheader("📅 Daily Average Energy Usage")
    daily_avg = df[energy_col].resample("D").mean()
    st.line_chart(daily_avg)

    # ML prediction
    st.subheader("🧠 Machine Learning Prediction")

    # Only show usable numeric features with enough data
    min_valid_rows = 50
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    usable_features = [col for col in numeric_cols if col != energy_col and df[col].notna().sum() >= min_valid_rows]

    feature_cols = st.multiselect("📊 Select features to predict energy usage", usable_features)

    if feature_cols:
        X = df[feature_cols]
        y = df[energy_col]

        # Impute missing values in features
        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

        # Drop rows with missing target
        valid_rows = y.notna()
        X_imputed = X_imputed.loc[valid_rows]
        y = y.loc[valid_rows]

        st.write(f"📊 Valid samples available for training: {len(X_imputed)}")

        if len(X_imputed) == 0:
            st.error("❌ No valid rows available after imputing and filtering.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)

            st.success(f"✅ Model trained! Mean Absolute Error: {mae:.2f}")

            st.subheader("📊 Actual vs Predicted Energy Usage")
            results = pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}, index=y_test.index)
            st.dataframe(results.head())
            st.line_chart(results.head(100))
    else:
        st.info("📌 Select at least one feature to enable ML prediction.")

    # Energy-saving insight
    st.subheader("💡 Energy Saving Insight")
    avg_energy = df[energy_col].mean()
    high_usage = df[df[energy_col] > avg_energy * 1.3]

    if not high_usage.empty:
        st.warning("⚠ High energy usage periods detected. Consider shifting usage to off-peak times.")
    else:
        st.info("✅ Your energy usage appears efficient!")

else:
    st.info("📌 Upload a CSV file to get started.")
