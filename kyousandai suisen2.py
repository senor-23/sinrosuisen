import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ===============================
# データ読み込み
# ===============================
df = pd.read_excel("excel2.xlsx")

# ===============================
# 列定義
# ===============================
bunkei_courses = [
    '経済/経済','経営/マネジメント','法/法律','法/法政策',
    '現代社会/現代社会','現代社会/健康スポーツ社会',
    '国際関係/国際関係',
    '外国語/英語','外国語/ヨーロッパ言語','外国語/アジア言語',
    '文化/文化構想','文化/京都文化','文化/文化観光'
]

rikei_courses = [
    '理/数理科','理/物理科','理/宇宙物理・気象',
    '情報理工/情報理工',
    '生命科/先端生命科','生命科/産業生命科'
]

course_columns = bunkei_courses + rikei_courses

interest_columns = ['旅行','読書','音楽','スポーツ','映画・ドラマ','ゲーム','アニメ・漫画']
meta_columns = ['性別','文理','偏差値']
character_columns = [
    'ISTJ(ロジスティシャン)','ISFJ(擁護者)','INFJ(提唱者)','INTJ(建築家)',
    'ISTP(巨匠)','ISFP(冒険家)','INFP(仲介者)','INTP(論理学者)',
    'ESTP(起業家)','ESFP(エンターテイナー)','ENFP(運動家)','ENTP(討論者)',
    'ESTJ(幹部)','ESFJ(領事)','ENFJ(主人公)','ENTJ(指揮官)'
]
subject_columns = ['国語','数学','英語','理科','社会']

# ===============================
# UI：重み調整
# ===============================
st.sidebar.title("⚙ 重み調整")
interest_w = st.sidebar.slider("興味の重み", 0.5, 5.0, 3.0)
subject_w  = st.sidebar.slider("得意科目の重み", 0.5, 8.0, 5.0)
mbti_w     = st.sidebar.slider("MBTIの重み", 0.5, 5.0, 2.0)
meta_w     = st.sidebar.slider("属性の重み", 0.1, 2.0, 1.0)

# ===============================
# データ分割
# ===============================
course_df = df[course_columns]

features_df = df[
    interest_columns + meta_columns + character_columns + subject_columns
].copy()

features_df[interest_columns]  *= interest_w
features_df[subject_columns]   *= subject_w
features_df[character_columns] *= mbti_w
features_df[meta_columns]      *= meta_w

# ===============================
# 学部平均との差（バイアス）
# ===============================
faculty_map = {
    '経済': ['経済/経済'],
    '経営': ['経営/マネジメント'],
    '法': ['法/法律','法/法政策'],
    '現代社会': ['現代社会/現代社会','現代社会/健康スポーツ社会'],
    '国際関係': ['国際関係/国際関係'],
    '外国語': ['外国語/英語','外国語/ヨーロッパ言語','外国語/アジア言語'],
    '文化': ['文化/文化構想','文化/京都文化','文化/文化観光'],
    '理': ['理/数理科','理/物理科','理/宇宙物理・気象'],
    '情報理工': ['情報理工/情報理工'],
    '生命科学': ['生命科/先端生命科','生命科/産業生命科']
}

faculty_mean = {
    faculty: course_df[cols].mean().mean()
    for faculty, cols in faculty_map.items()
}

# ===============================
# 推薦関数
# ===============================
def recommend_courses(user_features, bunri, top_n=3):
    user_vec = np.array(user_features).reshape(1, -1)
    user_vec /= (np.linalg.norm(user_vec) + 1e-8)

    X = features_df.values
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    similarities = cosine_similarity(user_vec, X)[0]

    top_k = 50
    top_idx = np.arg_
