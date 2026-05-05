import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

@st.cache_data
def prepare_knn():
    # Load dataset
   # Pastikan nama folder sesuai dengan yang ada di GitHub kamu
    df = pd.read_csv('TugasDeployDataKaggle/Social_Network_Ads.csv')
    X = df[['Age', 'EstimatedSalary']]
    y = df['Purchased']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    
    return df, model, scaler, acc

def main():
    st.set_page_config(page_title="Social Ads K-NN", layout="centered")
    st.title("🎯 Klasifikasi Social Ads (K-NN)")

    try:
        df, model, scaler, acc = prepare_knn()

        age = st.sidebar.slider("Umur", int(df.Age.min()), int(df.Age.max()), 30)
        salary = st.sidebar.slider("Gaji ($)", int(df.EstimatedSalary.min()), int(df.EstimatedSalary.max()), 50000)

        if st.button("Prediksi"):
            user_input = scaler.transform([[age, salary]])
            prediction = model.predict(user_input)[0]
            
            if prediction == 1:
                st.success("Hasil: Pelanggan akan membeli!")
            else:
                st.warning("Hasil: Pelanggan tidak membeli.")
            st.info(f"Akurasi Model: {acc*100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()