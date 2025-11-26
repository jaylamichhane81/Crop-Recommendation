import streamlit as st
import pandas as pd
import numpy as np

# sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Recommendation App", layout="wide")
st.title("🌾 Crop Recommendation System (ML Model Inside App)")
st.write("This app predicts the best crop based on soil and climate conditions.")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

df = load_data()

st.subheader("📌 Dataset Overview")
st.dataframe(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

numeric_cols = ['N','P','K','temperature','humidity','ph','rainfall']
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

X = df.drop('label', axis=1)
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Best Model (Random Forest)
# -------------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# Show Evaluation Metrics
# -------------------------------
with st.expander("📊 Model Performance"):
    st.write(f"### ✅ Accuracy: **{accuracy:.2f}**")
    st.write("### Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.write("### Confusion Matrix:")
    st.dataframe(confusion_matrix(y_test, y_pred))

# -------------------------------
# Prediction UI
# -------------------------------
st.subheader("🔮 Predict the Best Crop")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 200, 50)
    P = st.number_input("Phosphorus (P)", 0, 200, 50)
    K = st.number_input("Potassium (K)", 0, 200, 50)

with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 100.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)

with col3:
    ph = st.number_input("pH Value", 0.0, 14.0, 6.5)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)

if st.button("🌱 Recommend Crop"):
    user_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # scale numeric inputs
    user_data_scaled = scaler.transform(user_data)

    # prediction
    prediction = model.predict(user_data_scaled)[0]
    crop_name = encoder.inverse_transform([prediction])[0]

    st.success(f"### 🌾 Recommended Crop: **{crop_name}**")

