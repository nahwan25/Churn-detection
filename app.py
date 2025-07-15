import streamlit as st
import pandas as pd
import joblib
from preprocess_input import preprocess_input

# Load model
try:
    model = joblib.load('artifacts/model_pipeline.pkl')
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")

st.title("📊 Prediksi Churn Pelanggan")
st.write("Silakan upload file CSV untuk memulai prediksi.")

uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Asli")
        st.write(df_raw.head())

        df_clean = preprocess_input(df_raw)
        preds = model.predict(df_clean)
        df_raw['Churn Prediction'] = preds

        st.subheader("✅ Hasil Prediksi")
        st.write(df_raw[['customerID', 'Churn Prediction']] if 'customerID' in df_raw.columns else df_raw)

        # Download button
        csv = df_raw.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Hasil", csv, "hasil_prediksi.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Gagal memproses data: {e}")
