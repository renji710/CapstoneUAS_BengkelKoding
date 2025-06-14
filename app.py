import streamlit as st
import pandas as pd
import numpy as np
import joblib

#Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Tingkat Obesitas",
    page_icon="🧑‍⚕️",
    layout="centered",
    initial_sidebar_state="auto"
)

#Fungsi untuk Memuat Model dan Scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('obesity_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

#Fungsi untuk Prediksi
def predict_obesity(model, scaler, input_data):
    original_cols = ['Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP',
                     'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight', 'FAF',
                     'TUE', 'CAEC', 'MTRANS']
    
    selected_features = ['Weight', 'Height', 'FCVC', 'Age', 'Gender', 'TUE', 'CH2O', 'FAF']

    default_values_dict = {'CALC': 0.3333333333333333, 'FAVC': 1.0, 'NCP': 0.625, 'SCC': 0.0, 'SMOKE': 0.0, 'family_history_with_overweight': 1.0, 'CAEC': 0.3333333333333333, 'MTRANS': 0.75}

    
    full_input_df = pd.DataFrame([input_data])
    
    for col, default_val in default_values_dict.items():
        if col not in full_input_df.columns:
            full_input_df[col] = default_val
            
    full_input_df = full_input_df[original_cols]

    scaled_input_array = scaler.transform(full_input_df)
    scaled_input_df = pd.DataFrame(scaled_input_array, columns=original_cols)

    final_input = scaled_input_df[selected_features]

    prediction = model.predict(final_input)
    prediction_proba = model.predict_proba(final_input)

    return prediction[0], prediction_proba

#UI Streamlit
st.title("Prediksi Tingkat Obesitas 🧑‍⚕️")
st.write(
    "Aplikasi ini memprediksi tingkat obesitas berdasarkan 8 fitur utama. "
    "Masukkan data Anda di bawah ini untuk melihat hasilnya."
)

#Muat model dan scaler
model, scaler = load_model_and_scaler()

target_mapping = {
    0: 'Insufficient_Weight', 1: 'Normal_Weight', 2: 'Obesity_Type_I',
    3: 'Obesity_Type_II', 4: 'Obesity_Type_III', 5: 'Overweight_Level_I',
    6: 'Overweight_Level_II'
}

#Form untuk input pengguna
with st.form("obesity_prediction_form"):
    st.header("Masukkan Data Diri Anda:")

    st.subheader("Informasi Dasar")
    col1, col2 = st.columns(2)

    with col1:
        Weight = st.number_input(
            label='⚖️ Berat Badan (kg)', 
            min_value=30.0, 
            max_value=200.0, 
            value=70.0, 
            step=0.1,
            help="Masukkan berat badan Anda dalam kilogram."
        )
        Height = st.number_input(
            label='📏 Tinggi Badan (meter)', 
            min_value=1.0, 
            max_value=2.5, 
            value=1.70, 
            step=0.01,
            help="Masukkan tinggi badan Anda dalam format meter (contoh: 1.75)."
        )
        
    with col2:
        Age = st.number_input(
            label='🎂 Usia (tahun)', 
            min_value=14, 
            max_value=110, 
            value=25,
            help="Masukkan usia Anda saat ini."
        )
        Gender = st.selectbox(
            label='🚻 Jenis Kelamin', 
            options=[('Pria', 1), ('Wanita', 0)], 
            format_func=lambda x: x[0],
            help="Pilih jenis kelamin Anda."
        )[1]

    st.subheader("Aktivitas & Gaya Hidup")
    col3, col4 = st.columns(2)

    with col3:
        fcvc_options = {1: 'Jarang', 2: 'Kadang-kadang', 3: 'Selalu'}
        fcvc_label = st.selectbox(
            label='🥦 Seberapa sering Anda makan sayuran?',
            options=list(fcvc_options.keys()),
            format_func=lambda x: fcvc_options[x],
            index=1,
            help="Pilih frekuensi Anda mengonsumsi sayuran dalam makanan Anda."
        )
        FCVC = fcvc_label

        faf_options = {0: 'Tidak Pernah', 1: '1-2 hari/minggu', 2: '2-4 hari/minggu', 3: '4-5 hari/minggu'}
        faf_label = st.selectbox(
            label='🏃‍♀️ Seberapa sering Anda beraktivitas fisik?',
            options=list(faf_options.keys()),
            format_func=lambda x: faf_options[x],
            index=1,
            help="Aktivitas fisik tidak termasuk aktivitas sehari-hari seperti berjalan di rumah."
        )
        FAF = faf_label
        
    with col4:
        tue_options = {0: '0-2 jam', 1: '3-5 jam', 2: 'Lebih dari 5 jam'}
        tue_label = st.selectbox(
            label='📱 Berapa lama Anda menggunakan gadget?',
            options=list(tue_options.keys()),
            format_func=lambda x: tue_options[x],
            index=1,
            help="Ini termasuk ponsel, video game, televisi, komputer, dll."
        )
        TUE = tue_label

        ch2o_options = {1: 'Kurang dari 1 Liter', 2: 'Antara 1-2 Liter', 3: 'Lebih dari 2 Liter'}
        ch2o_label = st.selectbox(
            label='💧 Berapa banyak air yang Anda minum setiap hari?',
            options=list(ch2o_options.keys()),
            format_func=lambda x: ch2o_options[x],
            index=1,
            help="Pilih jumlah konsumsi air harian Anda."
        )
        CH2O = ch2o_label
    
    st.write("")

    submit_button = st.form_submit_button(label='Prediksi Sekarang!',use_container_width=True)

#Logika submit, prediksi, dan hasil
if submit_button:
    input_data = {
        'Weight': Weight,
        'Height': Height,
        'FCVC': FCVC,
        'Age': Age,
        'Gender': Gender,
        'TUE': TUE,
        'CH2O': CH2O,
        'FAF': FAF
    }
    
    prediction_code, prediction_proba = predict_obesity(model, scaler, input_data)
    
    prediction_label = target_mapping[prediction_code]

    st.subheader("Hasil Prediksi Anda:")
    
    st.metric(label="Tingkat Obesitas", value=prediction_label.replace('_', ' '))
    
    st.write("---")
    st.subheader("Detail Probabilitas Prediksi:")
    
    proba_df = pd.DataFrame({
        'Kategori': [label.replace('_', ' ') for label in target_mapping.values()],
        'Probabilitas': prediction_proba[0]
    }).sort_values(by='Probabilitas', ascending=False)
    
    st.bar_chart(proba_df.set_index('Kategori'))

    st.info(
        "**Disclaimer:** Hasil prediksi ini didasarkan pada model machine learning dan "
        "tidak boleh dianggap sebagai diagnosis medis. Silakan berkonsultasi dengan profesional kesehatan."
    )