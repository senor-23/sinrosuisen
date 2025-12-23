import pandas as pd


def load_and_preprocess(path):
    df = pd.read_excel("京産大　架空データbyチャッピー　1500.xlsx", sheet_name="Sheet1")

    feature_cols = [
        "性別", "文理", "MBTI", "都道府県", "得意科目"
    ]
    target_cols = ["学部", "学科"]

    X = df[feature_cols + ["偏差値"]]
    y = df[target_cols]

    categorical_cols = feature_cols
    numeric_cols = ["偏差値"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    X_encoded = preprocessor.fit_transform(X)

    return X_encoded, y, preprocessor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def recommend_faculty(
    X_encoded, y, user_vector, k=20
):
    similarities = cosine_similarity(user_vector, X_encoded)[0]
    top_k_idx = similarities.argsort()[-k:][::-1]

    similar_students = y.iloc[top_k_idx]

    result = (
        similar_students
        .value_counts()
        .reset_index(name="count")
    )

    return result
import streamlit as st
import pandas as pd
from src.preprocess import load_and_preprocess
from src.recommender import recommend_faculty

st.title("京都産業大学 学部推薦システム")

# データ読み込み
X_encoded, y, preprocessor = load_and_preprocess(
    "data/kyosan_dummy_data.xlsx"
)

# 入力UI
gender = st.selectbox("性別", ["男性", "女性"])
bunri = st.selectbox("文理", ["文系", "理系"])
mbti = st.selectbox("MBTI", ["INTJ","INFP","ENTP","ESFJ"])
pref = st.selectbox("出身地", ["京都府","大阪府","兵庫県"])
subject = st.selectbox("得意科目", ["国語","数学","英語","理科","社会"])
hensachi = st.slider("偏差値", 35, 75, 55)

if st.button("おすすめ学部を見る"):
    user_df = pd.DataFrame([{
        "性別": gender,
        "文理": bunri,
        "MBTI": mbti,
        "都道府県": pref,
        "得意科目": subject,
        "偏差値": hensachi
    }])

    user_vec = preprocessor.transform(user_df)
    rec = recommend_faculty(X_encoded, y, user_vec)

    st.subheader("あなたにおすすめの学部・学科")
    st.dataframe(rec.head(5))

