import streamlit as st
import pickle
import numpy as np

# import the module
pipe = pickle.load(open('pipe1.pkl', 'rb'))
df = pickle.load(open('df1.pkl', 'rb'))

st.title("💻 Laptop Price Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type
laptop_type = st.selectbox('Type', df['TypeName'].unique())


# RAM
ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screensize = st.number_input('Screen Size (in inches)')

# resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080','1366x768','1600x900','3840x2160','3200x1880',
    '2888x1800','2560x1680','2560x1440','2384x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD and SSD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # convert categorical features
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # resolution -> PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screensize

    # query
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)
    query = query.reshape(1, -1)

    # prediction
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"The predicted price of the configuration is ₹ {predicted_price:,}")
