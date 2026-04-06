import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# PAGE CONFIG

st.set_page_config(page_title="Bank Churn Predictor", layout="wide")

st.title("💳 Bank Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn using a trained CNN model.")

# LOAD DATA (for encoders + scaling)

@st.cache_data
def load_data():
    df = pd.read_csv("Bank Chunners.csv")
    df = df.iloc[:, :-2]
    df.drop("CLIENTNUM", axis=1, inplace=True)
    df.replace("Unknown", np.nan, inplace=True)

    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

df = load_data()

# ENCODING + SCALING SETUP

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop("Attrition_Flag", axis=1))

# MODEL DEFINITION (same as training)

class CNN1D(nn.Module):
    def __init__(self, input_size):
        super(CNN1D, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Dynamic size
        self._to_linear = None
        self._get_conv_output(input_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(self._to_linear, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def _get_conv_output(self, input_size):
        with torch.no_grad():
            x = torch.randn(1, 1, input_size)
            x = self.conv_layers(x)
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# LOAD / INITIALIZE MODEL

input_size = df.drop("Attrition_Flag", axis=1).shape[1]
model = CNN1D(input_size)

# NOTE: If you saved your trained model, load it here:
model.load_state_dict(torch.load("cnn_model.pth", map_location=torch.device('cpu')))
model.eval()

# USER INPUT UI

st.sidebar.header("Enter Customer Details")

def user_input():
    data = {}

    for col in df.drop("Attrition_Flag", axis=1).columns:

        if col in label_encoders:
            options = label_encoders[col].classes_
            data[col] = st.sidebar.selectbox(col, options)
        else:
            data[col] = st.sidebar.number_input(col, value=float(df[col].mean()))

    return pd.DataFrame([data])

input_df = user_input()

# PREPROCESS INPUT

def preprocess_input(input_df):
    df_copy = input_df.copy()

    # Ensure ALL columns match training data
    required_cols = df.drop("Attrition_Flag", axis=1).columns

    # Add missing columns
    for col in required_cols:
        if col not in df_copy.columns:
            df_copy[col] = df[col].mode()[0]  # default value

    # Reorder columns
    df_copy = df_copy[required_cols]

    # Apply encoding safely
    for col in label_encoders:
        if col in df_copy.columns:
            df_copy[col] = label_encoders[col].transform(df_copy[col])

    # Scale
    df_scaled = scaler.transform(df_copy)

    # Convert to tensor
    tensor = torch.tensor(df_scaled, dtype=torch.float32).unsqueeze(1)

    return tensor

# PREDICTION

if st.button("🔍 Predict Churn"):

    input_tensor = preprocess_input(input_df)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    prediction = np.argmax(probs)
    churn_prob = probs[1]

    # OUTPUT RESULTS

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to CHURN")
    else:
        st.success(f"✅ Customer is likely to STAY")

    st.write(f"**Churn Probability:** {churn_prob:.2%}")

    # BUSINESS INTERPRETATION
 
    st.subheader("Business Insight")

    if churn_prob > 0.7:
        st.write("🔴 High Risk: Immediate retention action recommended.")
    elif churn_prob > 0.4:
        st.write("🟠 Medium Risk: Monitor and engage customer.")
    else:
        st.write("🟢 Low Risk: Customer is stable.")

# FOOTER

st.markdown("---")
st.write("Input columns:", input_df.columns)
st.write("Expected columns:", df.drop("Attrition_Flag", axis=1).columns)
st.write("Built with Streamlit | CNN Model for Customer Churn Prediction")
