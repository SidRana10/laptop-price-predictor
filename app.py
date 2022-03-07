import streamlit as st
import pickle
import numpy as np

# import the model
pipe4 = pickle.load(open('pipe4.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])


#cpu
cpu = st.selectbox('CPU',df['cpu_name'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu brand'].unique())

os = st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    # query

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0


    query = np.array([company,type,ram,weight,touchscreen,ips,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1, 11)
    st.title("The predicted price of this Laptop is " + str(int(np.exp(pipe4.predict(query)[0]))))
