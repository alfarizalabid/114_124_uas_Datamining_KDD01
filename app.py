import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="California Housing Data Mining", layout="wide")

# Load model dan alat pendukung
@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    kmeans = joblib.load('kmeans.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, kmeans, scaler

model, kmeans, scaler = load_models()

# Load sebagian data untuk visualisasi
@st.cache_data
def load_data():
    return pd.read_csv('housing.csv').dropna()

df = load_data()

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", 
                        ["Home", "Dataset Overview", "Prediction", "Visualization", "About"])


# home
if menu == "Home":
    st.title("Prediksi Harga Rumah & Klastering Wilayah (California)")
    st.write("Aplikasi ini dibangun menggunakan algoritma **Random Forest** (Prediksi) dan **K-Means** (Clustering) untuk memprediksi harga rumah berdasarkan demografi dan lokasi di California.")
    
    st.info("""
    **Anggota Kelompok:**
    * Rizal Abid Alfarizi (24051214114)
    * Muhadzdzib lutfhi Hadid (24051214124)
    """)

# DATASET OVERVIEW 
elif menu == "Dataset Overview":
    st.title("Dataset Overview")
    st.write(f"Total data: {df.shape[0]} baris dan {df.shape[1]} kolom.")
    st.dataframe(df.head(10))
    
    st.subheader("Statistik Sederhana")
    st.write(df.describe())

# PREDICTION 
elif menu == "Prediction":
    st.title("Prediksi Harga Rumah")
    st.write("Masukkan spesifikasi rumah untuk melihat prediksi harga dan wilayah klaster.")

    col1, col2 = st.columns(2)
    with col1:
        longitude = st.number_input("Longitude", value=-122.23)
        latitude = st.number_input("Latitude", value=37.88)
        housing_median_age = st.number_input("Umur Bangunan (Tahun)", value=41.0)
        total_rooms = st.number_input("Total Ruangan", value=880.0)
        total_bedrooms = st.number_input("Total Kamar Tidur", value=129.0)
    with col2:
        population = st.number_input("Populasi Penduduk", value=322.0)
        households = st.number_input("Jumlah Kepala Keluarga", value=126.0)
        median_income = st.number_input("Pendapatan Menengah (x$10,000)", value=8.32)
        ocean_proximity = st.selectbox("Kedekatan dengan Laut", 
                                       ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

    if st.button("Proses Prediksi"):
        # (Catatan: Di sini nanti kita akan tambahkan logika preprocessing input pengguna
        # agar sesuai dengan format array X_train sebelum dimasukkan ke model.predict)
        st.success("Tombol berhasil ditekan! (Logika prediksi akan dihubungkan di tahap selanjutnya)")

# VISUALIZATION
elif menu == "Visualization":
    st.title("Visualisasi Hasil Analisis")
    
    st.subheader("Korelasi Pendapatan dan Harga Rumah")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='median_income', y='median_house_value', alpha=0.5, ax=ax)
    st.pyplot(fig)

# HALAMAN ABOUT 
elif menu == "About":
    st.title("Tentang Proyek")
    st.write("""
    **Metode yang Digunakan:**
    1. **K-Means Clustering:** Digunakan untuk membagi wilayah perumahan berdasarkan lokasi dan pendapatan.
    2. **Random Forest Regression:** Model machine learning tingkat lanjut untuk memprediksi harga rumah dengan akurasi tinggi.
    
    **Sumber Dataset:**
    Dataset California Housing dari Kaggle.
    """)