import streamlit as st
import pandas as pd
import os

# 필요한 패키지 설치
os.system('pip install matplotlib')

import matplotlib
matplotlib.use('Agg')  # To use a non-GUI backend
import matplotlib.pyplot as plt

# 파일 업로드
data_file = 'movies_2024.csv'
df = pd.read_csv(data_file)

# Streamlit 앱 설정
st.title('Budget vs Revenue Analysis')

# 데이터 불러오기
st.write("## Uploaded Dataset")
st.dataframe(df)

# budget과 revenue의 관계를 시각화
if 'budget' in df.columns and 'revenue' in df.columns:
    st.write("## Budget vs Revenue Scatter Plot")
    fig, ax = plt.subplots()
    ax.scatter(df['budget'], df['revenue'], alpha=0.5)
    ax.set_xlabel('Budget')
    ax.set_ylabel('Revenue')
    ax.set_title('Budget vs Revenue')
    st.pyplot(fig)
else:
    st.error("The dataset does not contain 'budget' or 'revenue' columns.")