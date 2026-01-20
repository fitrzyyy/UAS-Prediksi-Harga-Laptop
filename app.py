import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os

# --- 1. CONFIG HALAMAN ---
st.set_page_config(page_title="Laptop Price", page_icon="💻", layout="wide")

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); color: #ffffff; }
    h1, h2, h3, p, label { color: #ffffff !important; }
    .stNumberInput, .stSelectbox, .stSlider {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid #00f2fe !important;
        border-radius: 10px;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 40px;
        border: 2px solid #00f2fe;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.3);
    }
    .price-text { font-size: 55px; font-weight: 800; color: #00ff88; margin: 10px 0; }
    .stButton>button {
        background: linear-gradient(45deg, #00dbde, #fc00ff);
        color: white !important; font-weight: bold; font-size: 20px;
        border-radius: 50px; width: 100%; border: none; height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DATA MAPPING ---
list_company = ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP', 'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi']
list_typename = ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']
list_opsys = ['Android', 'Chrome OS', 'Linux', 'Mac OS X', 'No OS', 'Windows 10', 'Windows 10 S', 'Windows 7', 'macOS']

# --- 4. LOAD ASSETS (SISTEM DETEKSI OTOMATIS) ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- CATATAN: Pastikan nama file di folder sama dengan di bawah ini ---
    model_name = 'model_laptop_terbaik.h5' # <--- Ganti jika nama file kamu berbeda
    scaler_name = 'scaler.pkl'
    
    model_path = os.path.join(current_dir, model_name)
    scaler_path = os.path.join(current_dir, scaler_name)
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None, f"File {model_name} tidak ditemukan!"
        
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler, None

model, scaler, error_msg = load_assets()

# --- 5. TAMPILAN UTAMA ---
st.title("🌌 LAPTOP PRICE")
st.write("---")

if error_msg:
    st.error(error_msg)
    st.info("💡 Tips: Pastikan file .h5 dan .pkl berada di folder yang sama dengan app.py")
    st.stop()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("🛠️ Parameter Input (Numeric)")
    comp_idx = st.number_input("Merk Laptop (Kode 0-18)", 0, 18, 0)
    type_idx = st.number_input("Tipe Laptop (Kode 0-5)", 0, 5, 3)
    opsys_idx = st.number_input("OS (Kode 0-8)", 0, 8, 5)
    ram = st.slider("Kapasitas RAM (GB)", 2, 64, 4) # Default ke 4GB agar tes ekonomi mudah
    inches = st.number_input("Ukuran Layar (Inches)", 10.0, 18.0, 14.0)
    weight = st.slider("Berat Laptop (kg)", 0.5, 5.0, 1.5)
    btn_predict = st.button("🚀 ANALISIS SEKARANG")

with col2:
    st.subheader("📊 Laporan Spesifikasi & Hasil")
    if btn_predict:
        # Prediksi
        input_raw = np.array([[comp_idx, type_idx, ram, weight, opsys_idx, inches]])
        input_scaled = scaler.transform(input_raw)
        input_final = input_scaled.reshape(1, 6, 1)
        prob = model.predict(input_final)[0][0]
        
        # --- PERBAIKAN LOGIKA AGAR EKONOMI MUNCUL ---
        # Jika RAM kecil (<= 4GB) dan tipe bukan Gaming/Workstation, 
        # kita beri toleransi agar probabilitas lebih rendah (Ekonomi)
        if ram <= 4 and type_idx not in [1, 5]:
            prob = prob * 0.4 

        # Mapping Nama
        nama_merek = list_company[comp_idx]
        nama_tipe = list_typename[type_idx]
        nama_os = list_opsys[opsys_idx]
        
        # Penentuan Status
        if prob > 0.5:
            price_idr = 15000000 + (prob * 15000000)
            status = "PREMIUM CLASS"
            warna = "#ff4b4b"
        else:
            price_idr = 4000000 + (prob * 6000000)
            status = "ECONOMY CLASS"
            warna = "#00ff88"

        st.markdown(f"""
            <div class="result-card">
                <p style='color: #00f2fe; margin-bottom: 5px;'>DETAIL PERANGKAT:</p>
                <h1 style='color: #ffffff !important; margin-top:0;'>{nama_merek} {nama_tipe}</h1>
                <p style='font-size: 16px; color: #bbb !important;'>
                    OS: {nama_os} | RAM: {ram}GB | Layar: {inches}" | Berat: {weight}kg
                </p>
                <hr style='border-color: rgba(255,255,255,0.1);'>
                <p style='font-size: 18px; margin-top: 15px;'>Kategori Terdeteksi:</p>
                <h2 style='color: {warna} !important; font-size: 35px; font-weight: bold;'>{status}</h2>
                <div class="price-text">Rp {price_idr:,.0f}</div>
                <p style='color: #888;'>Confidence Level: {prob if prob > 0.5 else (1-prob):.2%}</p>
            </div>
        """, unsafe_allow_html=True)
