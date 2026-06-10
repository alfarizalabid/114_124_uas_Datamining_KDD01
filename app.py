import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="California Housing Data Mining", layout="wide")

@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    kmeans = joblib.load('kmeans.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, kmeans, scaler

model, kmeans, scaler = load_models()

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
        st.success("Tombol berhasil ditekan! (Logika prediksi akan dihubungkan di tahap selanjutnya)")

# VISUALIZATION
elif menu == "Visualization":
    st.title("Visualisasi Hasil Analisis")
    
    st.subheader("Korelasi Pendapatan dan Harga Rumah")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='median_income', y='median_house_value', alpha=0.5, ax=ax)
    st.pyplot(fig)

# ABOUT 
elif menu == "About":
    st.title("Tentang Proyek & Metodologi")
st.markdown("""
Aplikasi Web Prediksi Harga Rumah California ini dibangun menggunakan arsitektur **Hybrid Machine Learning** dua tahap (*Two-Stage Pipeline*). Sistem ini mengombinasikan pembelajaran tanpa pengawasan (*Unsupervised Learning*) 
dan pembelajaran terbimbing (*Supervised Learning*) untuk mengatasi kendala heterogenitas spasial pada data real estat.
""")

st.markdown("---")

with st.expander("1. Algoritma K-Means Clustering (Segmentasi Wilayah)", expanded=True):
    st.markdown("""
    **K-Means Clustering** adalah algoritma *Unsupervised Learning* yang berfungsi untuk mempartisi data ke dalam sejumlah $k$ kelompok (klaster). Dalam proyek ini, **K-Means ($k=3$)** digunakan sebagai agen dekomposisi spasial untuk memecah *California Housing Dataset* makro menjadi 3 sub-pasar (*submarkets*) yang homogen berdasarkan koordinat geografis dan demografi lokal.
    
    ### Karakteristik 3 Klaster Wilayah di Web Ini:
    * **Cluster 0 - Kawasan Pesisir Elit (*Coastal Premium Submarket*):** Wilayah yang terletak di sepanjang pantai barat California (seperti San Francisco dan San Diego). Penduduknya memiliki tingkat pendapatan menengah tinggi (*Median Income* > 5.0 atau > \$50,000/tahun) dengan harga rumah rata-rata tertinggi.
    * **Cluster 1 - Kawasan Pedalaman & Agraris (*Central Inland Submarket*):** Mencakup wilayah lembah tengah (*Central Valley*, seperti Sacramento dan Fresno). Karakteristiknya adalah kepadatan penduduk rendah, pendapatan kurva bawah, dan struktur harga rumah yang sangat ekonomis.
    * **Cluster 2 - Kawasan Urban Padat Penduduk (*Urban Dense Submarket*):** Dipicu secara sensitif oleh lonjakan volume *Population* dan *Households* (bisa > 5.000 jiwa per blok sensus), seperti area pusat kota Los Angeles. Dinamika harganya sangat dipengaruhi oleh kelangkaan lahan akibat kepadatan penduduk.
    """)

with st.expander("2. Algoritma Random Forest Regression (Mesin Prediksi Harga)", expanded=True):
    st.markdown("""
    **Random Forest Regressor** adalah algoritma *Supervised Learning* berbasis pembelajaran ansambel (*Ensemble Learning*). Algoritma ini tidak hanya mengandalkan satu pohon keputusan (*Decision Tree*), melainkan membangun **100 pohon keputusan independen** (`n_estimators=100`) selama fase pelatihan.
    
    ### Mekanisme Kerja Model:
    1.  **Bootstrap Sampling:** Membuat subset data acak dari dataset asli untuk setiap pohon keputusan tunggal.
    2.  **Random Feature Selection:** Memilih sebagian fitur acak pada setiap percabangan (*node splitting*) untuk meminimalkan nilai *Mean Squared Error* (MSE).
    3.  **Agregasi Akhir (*Averaging*):** Output prediksi akhir harga rumah didapatkan dari nilai rata-rata aritmatika seluruh proyeksi yang dikeluarkan oleh 100 pohon individu tersebut. Pendekatan ini membuat model sangat tangguh terhadap masalah *overfitting* dan sangat akurat dalam menangani hubungan data non-linear.
    """)

with st.expander("3. Mengapa Menggunakan Pendekatan Hybrid?", expanded=True):
    st.markdown("""
    ### Sinergi Model Dua Tahap (*Two-Stage Pipeline*):
    Jika memprediksi harga rumah California menggunakan satu model regresi biasa, hasilnya akan rawan mengalami bias (*underfitting*). Hal ini karena model dipaksa untuk merata-ratakan pola harga rumah mewah di pantai dengan rumah murah di pedesaan.
    
    Melalui pendekatan **Hybrid K-Means + Random Forest**, aliran data diatur sebagai berikut:
    1.  **K-Means** bertugas menyaring heterogenitas makro regional dan mengelompokkan data ke dalam sub-pasar yang sejenis.
    2.  Label klaster tersebut dijadikan **fitur tambahan (fitur ke-14)** untuk memperkaya basis pengetahuan kontekstual model regresi.
    3.  **Random Forest** dapat memprediksi harga secara jauh lebih spesifik dan responsif karena pohon keputusan langsung tahu data masukan pengguna masuk ke klaster wilayah mana.
    
    ### Dokumentasi Dataset:
    * **Sumber Data:** Biro Sensus Amerika Serikat (*California Housing Dataset*).
    * **Volume Data:** 20.640 baris observasi spasial-demografi.
    """)