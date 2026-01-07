import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# データ読み込み
# ===============================
df = pd.read_excel("excel1.xlsx", sheet_name="Sheet1")

# ===============================
# 列定義
# ===============================
course_columns = [
    '経済/経済', '経営/マネジメント', '法/法律', '法/法政策',
    '現代社会/現代社会', '現代社会/健康スポーツ社会',
    '国際関係/国際関係', '外国語/英語', '外国語/ヨーロッパ言語',
    '外国語/アジア言語', '文化/文化構想', '文化/京都文化',
    '文化/文化観光', '理/数理科', '理/物理科',
    '理/宇宙物理・気象', '情報理工/情報理工',
    '生命科/先端生命科', '生命科/産業生命科'
]

interest_columns = ['旅行', '読書', '音楽', 'スポーツ', '映画・ドラマ', 'ゲーム', 'アニメ・漫画']
meta_columns = ['性別', '文理', '偏差値']
character_columns = [
    'ISTJ(ロジスティシャン)', 'ISFJ(擁護者)', 'INFJ(提唱者)', 'INTJ(建築家)',
    'ISTP(巨匠)', 'ISFP(冒険家)', 'INFP(仲介者)', 'INTP(論理学者)',
    'ESTP(起業家)', 'ESFP(エンターテイナー)', 'ENFP(運動家)', 'ENTP(討論者)',
    'ESTJ(幹部)', 'ESFJ(領事)', 'ENFJ(主人公)', 'ENTJ(指揮官)'
]
subject_columns = ['国語', '数学', '英語', '理科', '社会']

# ===============================
# DataFrame 分割
# ===============================
course_df = df[course_columns]
features_df = df[
    interest_columns + meta_columns + character_columns + subject_columns
].copy()

# ===============================
# 重み設定（重要）
# ===============================
interest_w = 3.0
subject_w = 3.0
mbti_w = 1.5
meta_w = 1.0

# ===============================
# 学習データ側に重み付け
# ===============================
features_df[interest_columns] *= interest_w
features_df[subject_columns] *= subject_w
features_df[character_columns] *= mbti_w
features_df[meta_columns] *= meta_w

# ===============================
# 推薦関数
# ===============================
def recommend_courses(user_features, course_df, features_df, top_n=5):
    assert len(user_features) == features_df.shape[1]

    user_vec = np.array(user_features).reshape(1, -1)

    similarities = cosine_similarity(
        user_vec, features_df.values
    )[0]

    sim_sum = similarities.sum()
    if sim_sum == 0:
        return pd.Series(
            np.zeros(course_df.shape[1]),
            index=course_df.columns
        )

    scores = np.dot(similarities, course_df.values) / sim_sum

    return (
        pd.Series(scores, index=course_df.columns)
        .sort_values(ascending=False)
        .head(top_n)
    )

# ===============================
# Streamlit UI
# ===============================
st.title("京産大 進路推薦システム")
st.write("あなたの興味・特徴をもとに、学科を推薦します。")

user_features = []

# 興味
st.subheader("1. 興味・関心")
for col in interest_columns:
    val = st.checkbox(col)
    user_features.append((1 if val else 0) * interest_w)

# 属性
st.subheader("2. 基本情報")
gender = st.selectbox("性別", ["男性", "女性"])
bunri = st.selectbox("文理", ["文系", "理系"])
hensachi = st.slider("偏差値（目安）", 35, 70, 50)

user_features += [
    (0 if gender == "男性" else 1) * meta_w,
    (0 if bunri == "文系" else 1) * meta_w,
    (hensachi / 100) * meta_w
]

# MBTI
st.subheader("3. MBTI")
mbti = st.selectbox("MBTI", character_columns)
for col in character_columns:
    user_features.append((1 if mbti == col else 0) * mbti_w)

# 得意科目
st.subheader("4. 得意科目")
kamoku = st.selectbox("得意科目", subject_columns)
for col in subject_columns:
    user_features.append((1 if kamoku == col else 0) * subject_w)

# ===============================
# 推薦実行
# ===============================
if st.button("進路を推薦する"):
    recs = recommend_courses(user_features, course_df, features_df, top_n=5)
    st.subheader("あなたにおすすめの学科")

    for i, (name, score) in enumerate(recs.items(), 1):
        st.write(f"{i}. {name}（スコア: {score:.2f}）")
