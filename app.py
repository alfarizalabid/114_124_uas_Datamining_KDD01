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
    st.write("Isi informasi rumah di bawah ini. Semua field wajib diisi sebelum menekan tombol prediksi.")

    KOTA_CALIFORNIA = {
        "Los Angeles":    {"longitude": -118.24, "latitude": 34.05,  "ocean_proximity": "NEAR OCEAN"},
        "San Francisco":  {"longitude": -122.42, "latitude": 37.77,  "ocean_proximity": "NEAR BAY"},
        "San Diego":      {"longitude": -117.16, "latitude": 32.72,  "ocean_proximity": "NEAR OCEAN"},
        "Sacramento":     {"longitude": -121.49, "latitude": 38.58,  "ocean_proximity": "INLAND"},
        "San Jose":       {"longitude": -121.89, "latitude": 37.34,  "ocean_proximity": "<1H OCEAN"},
        "Fresno":         {"longitude": -119.77, "latitude": 36.74,  "ocean_proximity": "INLAND"},
        "Long Beach":     {"longitude": -118.19, "latitude": 33.77,  "ocean_proximity": "<1H OCEAN"},
    }

    with st.expander("Lokasi Properti", expanded=True):
        kota_dipilih = st.selectbox(
            "Pilih Area / Kota Representatif",
            options=list(KOTA_CALIFORNIA.keys()),
            help="Pilih kota yang paling mendekati lokasi properti. "
                 "Nilai koordinat dan kedekatan laut akan terisi otomatis."
        )
        longitude        = KOTA_CALIFORNIA[kota_dipilih]["longitude"]
        latitude         = KOTA_CALIFORNIA[kota_dipilih]["latitude"]
        ocean_proximity  = KOTA_CALIFORNIA[kota_dipilih]["ocean_proximity"]

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Longitude", f"{longitude}")
        info_col2.metric("Latitude",  f"{latitude}")
        info_col3.metric("Kedekatan Laut", ocean_proximity)

    with st.expander("Kondisi Bangunan", expanded=True):
        housing_median_age = float(st.slider(
            "Umur Bangunan (Tahun)",
            min_value=1, max_value=50, value=20, step=1,
            help="Perkiraan usia rata-rata bangunan di blok perumahan tersebut. "
                 "Semakin tua bangunan, nilainya biasanya lebih rendah."
        ))

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            total_rooms = st.number_input(
                "Total Ruangan",
                min_value=1.0, value=880.0, step=50.0,
                help="Jumlah total seluruh ruangan (kamar tidur, ruang tamu, dapur, dll.) "
                     "di semua unit dalam satu blok perumahan."
            )
        with b_col2:
            total_bedrooms = st.number_input(
                "Total Kamar Tidur",
                min_value=1.0, value=129.0, step=10.0,
                help="Jumlah total khusus kamar tidur di semua unit dalam satu blok. "
                     "Nilainya harus lebih kecil dari Total Ruangan."
            )

    with st.expander("Data Demografis Wilayah", expanded=True):
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            population = st.number_input(
                "Populasi Penduduk",
                min_value=1.0, value=322.0, step=50.0,
                help="Total jumlah penduduk yang tinggal di blok perumahan tersebut."
            )
        with d_col2:
            households = st.number_input(
                "Jumlah Kepala Keluarga",
                min_value=1.0, value=126.0, step=10.0,
                help="Jumlah rumah tangga (keluarga) yang menempati unit-unit di blok ini. "
                     "Nilainya biasanya lebih kecil dari Populasi."
            )
        with d_col3:
            median_income = st.number_input(
                "Pendapatan Menengah",
                min_value=0.5, max_value=15.0, value=4.50, step=0.5,
                help="Pendapatan rata-rata rumah tangga dalam satuan puluhan ribu dolar. "
                     "Contoh: nilai 4.5 berarti $45,000/tahun, nilai 8.32 berarti $83,200/tahun."
            )
        st.caption("Tip: Pendapatan Menengah diisi dalam satuan ×$10,000. "
                   "Misal, pendapatan $50,000/tahun → isi angka **5.0**.")

    if st.button("Proses Prediksi"):
        with st.spinner("Memproses data..."):
            try:
                input_num = pd.DataFrame({
                    'longitude': [longitude],
                    'latitude': [latitude],
                    'housing_median_age': [housing_median_age],
                    'total_rooms': [total_rooms],
                    'total_bedrooms': [total_bedrooms],
                    'population': [population],
                    'households': [households],
                    'median_income': [median_income]
                })[scaler.feature_names_in_]  # pastikan urutan kolom sama persis

                scaled_arr = scaler.transform(input_num)
                scaled_df = pd.DataFrame(scaled_arr, columns=scaler.feature_names_in_)

                klaster_prediksi = int(kmeans.predict(scaled_df)[0])

                ohe_cols = [
                    'ocean_proximity_<1H OCEAN',
                    'ocean_proximity_INLAND',
                    'ocean_proximity_ISLAND',
                    'ocean_proximity_NEAR BAY',
                    'ocean_proximity_NEAR OCEAN',
                ]
                ohe_values = {col: 0 for col in ohe_cols}
                ohe_values[f'ocean_proximity_{ocean_proximity}'] = 1

                final_data = {**scaled_df.iloc[0].to_dict(), **ohe_values, 'cluster': klaster_prediksi}
                final_df = pd.DataFrame([final_data])[model.feature_names_in_]

                harga_prediksi = model.predict(final_df)[0]

                st.subheader("Hasil Prediksi")
                col_hasil1, col_hasil2 = st.columns(2)
                with col_hasil1:
                    st.info(f"Kategori Wilayah: **Cluster {klaster_prediksi}**")
                with col_hasil2:
                    st.success(f"Estimasi Nilai Rumah: **${harga_prediksi:,.2f}**")

            except Exception as e:
                st.error(f"Error pada sistem: {e}")

# VISUALIZATION
elif menu == "Visualization":
    st.title("Visualisasi Hasil Analisis")
    st.write("Eksplorasi dataset California Housing melalui berbagai grafik interaktif.")

    jenis_grafik = st.selectbox("Pilih Jenis Visualisasi:", 
                                ["Peta Persebaran Harga Rumah (Spasial)", 
                                 "Distribusi Harga Rumah (Histogram)", 
                                 "Hubungan Antar Fitur (Dinamis)"])

    st.markdown("---")

    if jenis_grafik == "Peta Persebaran Harga Rumah (Spasial)":
        st.subheader("Peta Persebaran Rumah di Wilayah California")
        st.write("Setiap titik mewakili blok perumahan. Warna merah menunjukkan area dengan harga rumah (Median House Value) yang lebih mahal.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(x=df['longitude'], y=df['latitude'], 
                             alpha=0.4, 
                             s=df['population']/100, 
                             c=df['median_house_value'], 
                             cmap=plt.get_cmap("jet"))
        
        fig.colorbar(scatter, ax=ax, label="Harga Rumah ($)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        st.pyplot(fig)

    elif jenis_grafik == "Distribusi Harga Rumah (Histogram)":
        st.subheader("Distribusi Nilai Tengah Harga Rumah")
        st.write("Grafik ini menunjukkan rentang harga rumah yang paling banyak mendominasi dataset.")
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['median_house_value'], bins=50, kde=True, color='teal', ax=ax)
        plt.xlabel("Harga Rumah ($)")
        plt.ylabel("Frekuensi (Jumlah Blok)")
        st.pyplot(fig)

    elif jenis_grafik == "Hubungan Antar Fitur (Dinamis)":
        st.subheader("Eksplorasi Hubungan Antar Variabel")
        st.write("Pilih fitur untuk sumbu X dan Y untuk melihat pola korelasinya secara langsung.")
        
        kolom_numerik = df.select_dtypes(include=np.number).columns
        
        col_x, col_y = st.columns(2)
        with col_x:
            sumbu_x = st.selectbox("Pilih Sumbu X:", kolom_numerik, index=7) 
        with col_y:
            sumbu_y = st.selectbox("Pilih Sumbu Y:", kolom_numerik, index=8)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=sumbu_x, y=sumbu_y, alpha=0.5, color='darkorange', ax=ax)
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