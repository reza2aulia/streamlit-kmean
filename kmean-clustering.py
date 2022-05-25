#!/usr/bin/env python
# coding: utf-8

# In[18]:


import warnings
import time
import webbrowser
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Country-data.csv')

# In[3]:


df_num = df.select_dtypes(include='number')

# In[19]:


df2 = df[['health', 'gdpp']].copy()

st.title("Interactive K-Means Clustering")

st.write("Berikut ini tampilan dataset untuk di Analisis clustering:")

st.dataframe(df2)


# In[6]:


# In[7]:


rscaled = RobustScaler()


# In[8]:


rscaled.fit(df2)


# In[9]:


df2_scaled = rscaled.transform(df2)


# In[10]:


df2_scaled = pd.DataFrame(df2_scaled)
df2_scaled.columns = df2.columns


# In[11]:


df_num = df.select_dtypes(include=np.number).columns
rscaler = RobustScaler()
df[df_num] = rscaler.fit_transform(df[df_num])

# sidebar
with st.sidebar:
    st.title('Reza Aulia')
    st.write("Perkenalkan Nama Saya Reza Aulia.")
    st.write('NPM : 2108207010004')
    st.write(
        "Hay berikut Project GUI menggunakan Streamlit.")
    nomor = st.slider('Pilih nomor untuk dilakukan cluster',
                      min_value=2, max_value=10)
    with st.spinner("Loading..."):
        time.sleep(5)
    st.success("Done!")
    st.write('Saya ingin terhubung dengan Anda di')
    url = 'https://www.linkedin.com/in/reza2aulia/'

    if st.button('LinkedIn'):
        webbrowser.open_new_tab(url)


# n = int(input('silahkan masukkan angka:'))
kmeans = KMeans(n_clusters=nomor, init='k-means++')
kmeans.fit(df2)

df2['label'] = kmeans.labels_


st.write("Jumlah clustering yang dipilih {} Cluster :".format(nomor))
st.write("Berikut Visualisasi dari Model K-Mean")
sns.scatterplot(data=df2, x='health', y='gdpp', hue='label', palette="deep")
sns.scatterplot(x=kmeans.cluster_centers_[
                :, 0], y=kmeans.cluster_centers_[:, 1], marker='*', s=500)
st.pyplot()
