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

# 重み適用
features_df[interest_columns] *= interest_w
features_df[subject_columns]  *= subject_w
features_df[character_columns]*= mbti_w
features_df[meta_columns]     *= meta_w

# ===============================
# 学部ごとの平均値（バイアス）
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

faculty_mean = {}
for faculty, cols in faculty_map.items():
    faculty_mean[faculty] = course_df[cols].mean(axis=1).mean()

# ===============================
# 推薦関数
# ===============================
def recommend_courses(user_features, bunri, top_n=3):
    user_vec = np.array(user_features).reshape(1, -1)
    user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)

    X = features_df.values
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    similarities = cosine_similarity(user_vec, X)[0]

    # 満足度で重み付け
    satisfaction = df["満足度"].values
    satisfaction_weight = satisfaction / satisfaction.max()
    weighted_sim = similarities * satisfaction_weight

    top_k = 50
    top_idx = np.argsort(weighted_sim)[-top_k:]
    top_sim = weighted_sim[top_idx]

    raw_score = (
        np.dot(top_sim, course_df.values[top_idx])
        / (top_sim.sum() + 1e-8)
    )

    score = pd.Series(raw_score, index=course_columns)

    # =========================
    # ★ 学部平均との差を引く（最重要）
    # =========================
    for faculty, cols in faculty_map.items():
        for col in cols:
            score[col] -= faculty_mean[faculty]

    # 文理フィルタ
    if bunri == "文系":
        score = score[bunkei_courses]
    else:
        score = score[rikei_courses]

    return score.sort_values(ascending=False).head(top_n)

# ===============================
# UI
# ===============================
st.title("🎓 京産大 進路推薦（ミスマッチ防止型）")

user_features = []

st.subheader("① 興味")
for col in interest_columns:
    user_features.append((1 if st.checkbox(col) else 0) * interest_w)

st.subheader("② 基本情報")
gender = st.selectbox("性別", ["男性","女性"])
bunri = st.selectbox("文理", ["文系","理系"])
hensachi = st.slider("偏差値", 35, 70, 50)

user_features += [
    (0 if gender=="男性" else 1)*meta_w,
    (0 if bunri=="文系" else 1)*meta_w,
    (hensachi/100)*meta_w
]

st.subheader("③ MBTI")
mbti = st.selectbox("MBTI", character_columns)
for col in character_columns:
    user_features.append((1 if col==mbti else 0)*mbti_w)

st.subheader("④ 得意科目")
kamoku = st.selectbox("得意科目", subject_columns)
for col in subject_columns:
    user_features.append((1 if col==kamoku else 0)*subject_w)

# ===============================
# 実行
# ===============================
if st.button("進路を推薦"):
    result = recommend_courses(user_features, bunri)

    st.subheader("おすすめ学科")
    for i, (name, score) in enumerate(result.items(), 1):
        st.markdown(f"### {i}. {name}")
        st.write(f"スコア: {score:.2f}")
        st.write("**理由：**")
        st.write("・あなたと似た学生の満足度が高い")
        st.write("・学部全体の人気に引っ張られていない")
