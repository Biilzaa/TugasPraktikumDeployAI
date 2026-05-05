import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_iris():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(url)
    X = df.drop('species', axis=1)
    y = df['species']
    model = RandomForestClassifier()
    model.fit(X, y)
    return df, model

def main():
    st.title("🌸 Klasifikasi Bunga Iris")
    df, model = load_iris()
    
    s_len = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    s_wid = st.slider("Sepal Width", 2.0, 5.0, 3.0)
    p_len = st.slider("Petal Length", 1.0, 7.0, 4.0)
    p_wid = st.slider("Petal Width", 0.1, 2.5, 1.0)
    
    if st.button("Prediksi"):
        pred = model.predict([[s_len, s_wid, p_len, p_wid]])
        st.success(f"Jenis Iris: {pred[0]}")

if __name__ == "__main__":
    main()