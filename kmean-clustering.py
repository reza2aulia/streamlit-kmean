import warnings
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)


warnings.filterwarnings("ignore")


df = pd.read_csv('Country-data.csv')
st.title("Interactive K-Means Clustering")

st.write("Data Indikator Kesejahteraan Negara")

st.dataframe(df)

# scalling
num_kolom = df.select_dtypes(include=np.number).columns.tolist()
scaler = MinMaxScaler()
df[num_kolom] = scaler.fit_transform(df[num_kolom])


# sidebar
with st.sidebar:
    st.title('Reza Aulia')

    # select box

    option1 = st.selectbox(
        'Pilihan 1',
        ([None] + num_kolom))

    st.write('You selected:', option1)

    if option1 is None:
        st.write("Pilihan 1 harus diisi")

    option2 = st.selectbox(
        'Pilihan 2',
        ([None] + num_kolom))

    st.write('You selected:', option2)
    if option2 is None:
        st.write("Pilihan 2 harus diisi")

    nomor = st.slider('Pilih jumlah cluster yang anda inginkan',
                      min_value=2, max_value=10)
    with st.spinner("Loading..."):
        time.sleep(5)
    st.success("Done!")
    st.write('Saya ingin terhubung dengan Anda di')
    url = 'https://www.linkedin.com/in/reza2aulia/'

    if st.button('LinkedIn'):
        webbrowser.open_new_tab(url)

# cluster
if option1 == option2:
    st.warning('Pilihan tidak boleh sama')
else:
    if option1 is None:
        st.warning('Pilihan 1 tidak boleh kosong')
    elif option2 is None:
        st.warning('Pilihan 2 tidak boleh kosong')
    else:
        st.write(
            "Pengelompokan Negara Berdasarkan Indikator Kesejahteraan Dengan Metode Unsupervised Learning-Clustering:")

        df_cluster = df[[option1, option2]].copy()
        kmeans = KMeans(n_clusters=nomor, init='k-means++')
        kmeans.fit(df_cluster)

        labels = kmeans.labels_
        centroid = kmeans.cluster_centers_

        sns.scatterplot(data=df_cluster, x=option1, y=option2, hue=labels)
        sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
            :, 1], marker='*', s=500)
        plt.show()
        st.pyplot()
