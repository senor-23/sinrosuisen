import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ======================
# データ読み込み
# ======================
df = pd.read_excel("京産大　架空データbyチャッピー　1500.xlsx", sheet_name="Sheet1")

course_columns = [
    '経済/経済','経営/マネジメント','法/法律','法/法政策',
    '現代社会/現代社会','現代社会/健康スポーツ社会',
    '国際関係/国際関係','外国語/英語','外国語/ヨーロッパ言語',
    '外国語/アジア言語','文化/文化構想','文化/京都文化',
    '文化/文化観光','理/数理科','理/物理科',
    '理/宇宙物理・気象','情報理工/情報理工',
    '生命科/先端生命科','生命科/産業生命科'
]

interest_columns = ['旅行','読書','音楽','スポーツ','映画・ドラマ','ゲーム','アニメ・漫画']
meta_columns = ['性別','文理','偏差値','満足度']
character_columns = [
    'ISTJ(ロジスティシャン)','ISFJ(擁護者)','INFJ(提唱者)','INTJ(建築家)',
    'ISTP(巨匠)','ISFP(冒険家)','INFP(仲介者)','INTP(論理学者)',
    'ESTP(起業家)','ESFP(エンターテイナー)','ENFP(運動家)',
    'ENTP(討論者)','ESTJ(幹部)','ESFJ(領事)','ENFJ(主人公)','ENTJ(指揮官)'
]
subject_columns = ['国語','数学','英語','理科','社会']
place_columns = [
    '北海道','青森','岩手','宮城','秋田','山形','福島','茨城','栃木','群馬',
    '埼玉','千葉','東京','神奈川','新潟','富山','石川','福井','山梨','長野',
    '岐阜','静岡','愛知','三重','滋賀','京都','大阪','兵庫','奈良','和歌山',
    '鳥取','島根','岡山','広島','山口','徳島','香川','愛媛','高知',
    '福岡','佐賀','長崎','熊本','大分','宮崎','鹿児島','沖縄'
]

course_df = df[course_columns]
features_df = df[
    interest_columns + meta_columns +
    character_columns + subject_columns + place_columns
]

# ======================
# 推薦関数
# ======================
def recommend_courses(user_features, course_df, features_df, top_n=3):
    user_vec = np.array(user_features).reshape(1, -1)
    sim = cosine_similarity(user_vec, features_df.values)[0]

    scores = np.dot(sim, course_df.values) / sim.sum()
    recs = pd.Series(scores, index=course_df.columns)

    return recs.sort_values(ascending=False).head(top_n)

# ======================
# Streamlit UI
# ======================
st.title("京産大 進路推薦システム")
st.write("あなたの特徴に近い学生のデータをもとに、満足度が高くなりやすい学科を推薦します。")

user_features = []

st.subheader("① 興味・関心")
for col in interest_columns:
    user_features.append(1 if st.checkbox(col) else 0)

st.subheader("② 基本情報")
gender = st.selectbox("性別", ["男性","女性"])
bunri = st.selectbox("文理", ["文系","理系"])
hensachi = st.slider("偏差値", 35, 70, 50)

user_features.append(1 if gender == "女性" else 0)
user_features.append(1 if bunri == "理系" else 0)
user_features.append(hensachi)
user_features.append(0)  # 満足度（ダミー）

st.subheader("③ MBTI")
mbti = st.selectbox("MBTI", character_columns)
for col in character_columns:
    user_features.append(1 if mbti == col else 0)

st.subheader("④ 得意科目")
kamoku = st.selectbox("得意科目", subject_columns)
for col in subject_columns:
    user_features.append(1 if kamoku == col else 0)

st.subheader("⑤ 出身地")
shusshin = st.selectbox("出身地", place_columns)
for col in place_columns:
    user_features.append(1 if shusshin == col else 0)

if st.button("進路を推薦する"):
    recs = recommend_courses(user_features, course_df, features_df)
    st.subheader("おすすめ学科")
    for i, (name, score) in enumerate(recs.items(), 1):
        st.write(f"{i}. {name}（予測スコア：{score:.2f}）")

