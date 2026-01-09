import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ===============================
# データ読み込み（満足度入り）
# ===============================
df = pd.read_excel("excel2.xlsx", sheet_name="Sheet1")

# ===============================
# 学科定義
# ===============================
bunkei_courses = [
    '経済/経済', '経営/マネジメント', '法/法律', '法/法政策',
    '現代社会/現代社会', '現代社会/健康スポーツ社会',
    '国際関係/国際関係',
    '外国語/英語', '外国語/ヨーロッパ言語', '外国語/アジア言語',
    '文化/文化構想', '文化/京都文化', '文化/文化観光'
]

rikei_courses = [
    '理/数理科', '理/物理科', '理/宇宙物理・気象',
    '情報理工/情報理工',
    '生命科/先端生命科', '生命科/産業生命科'
]

course_columns = bunkei_courses + rikei_courses

# ===============================
# 特徴量定義
# ===============================
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
alpha      = st.sidebar.slider("満足度 vs SVD", 0.0, 1.0, 0.8)

# ===============================
# データ分割
# ===============================
course_df = df[course_columns]  # ← 満足度（1〜10 or NaN）

features_df = df[
    interest_columns + meta_columns + character_columns + subject_columns
].copy()

# 重み反映（学習側）
features_df[interest_columns]  *= interest_w
features_df[subject_columns]   *= subject_w
features_df[character_columns] *= mbti_w
features_df[meta_columns]      *= meta_w

# ===============================
# SVD（補助）
# ===============================
svd = TruncatedSVD(n_components=5, random_state=42)
latent_user = svd.fit_transform(course_df.fillna(0))
latent_course = svd.components_

def svd_score():
    user_latent = latent_user.mean(axis=0)
    scores = np.dot(user_latent, latent_course)
    return pd.Series(scores, index=course_columns)

# ===============================
# 推薦ロジック（核心）
# ===============================
def recommend_courses(user_features, bunri, top_n=5):
    # ---- 類似度 ----
    user_vec = np.array(user_features).reshape(1, -1)
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)

    X = features_df.values
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    similarities = cosine_similarity(user_vec, X)[0]

    # ---- 満足度 × 類似度 ----
    sim = similarities.reshape(-1, 1)
    weighted_satisfaction = sim * course_df.values

    satisfaction_score = np.nanmean(weighted_satisfaction, axis=0)
    satisfaction_score = pd.Series(satisfaction_score, index=course_columns)

    # ---- SVD ----
    svd_scores = svd_score()

    # ---- ハイブリッド ----
    final_score = alpha * satisfaction_score + (1 - alpha) * svd_scores

    # ---- 文理フィルタ ----
    if bunri == "文系":
        final_score = final_score[bunkei_courses]
    else:
        final_score = final_score[rikei_courses]

    return final_score.sort_values(ascending=False).head(top_n)

# 学部ごとの平均を計算
faculty_mean = course_df.groupby(faculty_map, axis=1).mean()

# 学部平均との差を引く
course_debiased = course_centered.copy()
for course in course_centered.columns:
    faculty = faculty_map[course]
    course_debiased[course] -= faculty_mean[faculty]


# ===============================
# UI
# ===============================
st.title("🎓 京産大 進路推薦（満足度重視）")

user_features = []

st.subheader("① 興味")
for col in interest_columns:
    user_features.append((1 if st.checkbox(col) else 0) * interest_w)

st.subheader("② 基本情報")
gender = st.selectbox("性別", ["男性","女性"])
bunri  = st.selectbox("文理", ["文系","理系"])
hensachi = st.slider("偏差値", 35, 70, 50)

user_features += [
    (0 if gender=="男性" else 1) * meta_w,
    (0 if bunri=="文系" else 1) * meta_w,
    (hensachi / 100) * meta_w
]

st.subheader("③ MBTI")
mbti = st.selectbox("MBTI", character_columns)
for col in character_columns:
    user_features.append((1 if col == mbti else 0) * mbti_w)

st.subheader("④ 得意科目")
kamoku = st.selectbox("得意科目", subject_columns)
for col in subject_columns:
    user_features.append((1 if col == kamoku else 0) * subject_w)

# ===============================
# 実行
# ===============================
if st.button("進路を推薦"):
    result = recommend_courses(user_features, bunri, top_n=3)

    st.subheader("🌟 おすすめ学科")
    for i, (name, score) in enumerate(result.items(), 1):
        st.markdown(f"### {i}. {name}")
        st.write(f"スコア: {score:.2f}")
        st.write("**理由：**")
        st.write("・あなたと似た学生の満足度が高い")
        st.write("・興味・得意科目・傾向が一致")
