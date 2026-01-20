import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os

# --- 1. CONFIG HALAMAN ---
st.set_page_config(page_title="Laptop AI Predictor Pro", page_icon="💻", layout="wide")

# --- 2. CUSTOM CSS (FIX TEKS & VISUAL) ---
st.markdown("""
    <style>
    /* Background Utama Gelap Premium */
    .stApp {
        background: radial-gradient(circle, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Memastikan semua teks utama berwarna putih agar kelihatan */
    h1, h2, h3, p, label, .stMarkdown {
        color: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Style Input Box agar kontras */
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 5px;
    }

    /* Kartu Hasil Glassmorphism */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 30px;
        padding: 50px;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
        text-align: center;
    }

    /* Tombol Prediksi Neon */
    .stButton>button {
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        color: white !important;
        border: none;
        padding: 20px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.5);
        width: 100%;
    }
    
    .price-tag {
        font-size: 65px;
        font-weight: 900;
        color: #00f2fe;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.6);
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD ASSETS ---
@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'model_laptop_terbaik.h5')
    scaler_path = os.path.join(current_dir, 'scaler.pkl')
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = tf.keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

if model is None:
    st.error("❌ ERROR: File .h5 atau .pkl tidak ditemukan di folder!")
    st.stop()

# --- 4. DATA MAPPING ---
list_company = ['Acer', 'Apple', 'Asus', 'Chuwi', 'Dell', 'Fujitsu', 'Google', 'HP', 'Huawei', 'LG', 'Lenovo', 'MSI', 'Mediacom', 'Microsoft', 'Razer', 'Samsung', 'Toshiba', 'Vero', 'Xiaomi']
list_typename = ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']
list_opsys = ['Android', 'Chrome OS', 'Linux', 'Mac OS X', 'No OS', 'Windows 10', 'Windows 10 S', 'Windows 7', 'macOS']

# --- 5. LAYOUT UTAMA ---
st.write("# 🌌 LAPTOP PRICE")
st.write("Prediksi Masa Depan Harga Laptop Anda dengan Hybrid Deep Learning.")
st.write("---")

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("### 🛠️ Input Spesifikasi")
    
    # Grid Input
    brand = st.selectbox("Merk Laptop", list_company)
    l_type = st.selectbox("Tipe Laptop", list_typename)
    os_sys = st.selectbox("Sistem Operasi", list_opsys)
    
    c1, c2 = st.columns(2)
    with c1:
        ram = st.select_slider("RAM (GB)", options=[2, 4, 8, 12, 16, 32, 64], value=8)
    with c2:
        inch = st.number_input("Layar (Inches)", 10.0, 18.0, 15.6)
    
    weight = st.slider("Berat Laptop (kg)", 0.5, 5.0, 1.5)
    
    st.write(" ")
    btn_predict = st.button("🚀 ANALISIS SEKARANG")

with col_right:
    st.markdown("### 📊 Hasil Prediksi")
    if btn_predict:
        # Preprocessing & Prediction
        brand_enc = list_company.index(brand)
        type_enc = list_typename.index(l_type)
        os_enc = list_opsys.index(os_sys)
        
        input_raw = np.array([[brand_enc, type_enc, ram, weight, os_enc, inch]])
        input_scaled = scaler.transform(input_raw)
        input_final = input_scaled.reshape(1, 6, 1)
        
        prob = model.predict(input_final)[0][0]
        
        # Logika Harga Rupiah (Kurs 17.000)
        kurs = 17000
        median_base = 1000
        
        if prob > 0.5:
            price = (median_base + (prob * 1300)) * kurs
            st.markdown(f"""
                <div class="result-card">
                    <h2 style='color: #ff0055;'>💎 PREMIUM EDITION</h2>
                    <p style='color: #aaa;'>Estimasi Harga Pasar:</p>
                    <div class="price-tag">Rp {price:,.0f}</div>
                    <p style='color: #00f2fe;'>AI Confidence: {prob:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            price = (median_base - ((1-prob) * 600)) * kurs
            st.markdown(f"""
                <div class="result-card">
                    <h2 style='color: #00ff88;'>✅ ECONOMY EDITION</h2>
                    <p style='color: #aaa;'>Estimasi Harga Pasar:</p>
                    <div class="price-tag" style='color: #00ff88; text-shadow: 0 0 30px rgba(0, 255, 136, 0.4);'>
                        Rp {price:,.0f}
                    </div>
                    <p style='color: #00ff88;'>Prediksi: {(1-prob):.2%}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='padding: 50px; border: 2px dashed rgba(255,255,255,0.2); border-radius: 20px; text-align: center;'>
                <p style='color: rgba(255,255,255,0.4);'>Silakan masukkan spesifikasi di samping untuk memulai analisis AI.</p>
            </div>
        """, unsafe_allow_html=True)

st.write("---")
st.caption("UAS Deep Learning 2026 @ Project Fitri 220401075 | Powered by Hybrid CNN-GRU")