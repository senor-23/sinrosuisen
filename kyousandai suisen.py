import streamlit as st
import numpy as np
import pandas as pd

df = pd.read_excel("京産大　架空データbyチャッピー　1500.xlsx", sheet_name="Sheet1")

course_columns = ['経済/経済', '経営/マネジメント', '法/法律', '法/法政策', '現代社会/現代社会',
                  '現代社会/健康スポーツ社会', '国際関係/国際関係', '外国語/英語', '外国語/ヨーロッパ言語',
                  '外国語/アジア言語', '文化/文化構想', '文化/京都文化', '文化/文化観光', '理/数理科',
                  '理/物理科', '理/宇宙物理・気象', '情報理工/情報理工', '生命科/先端生命科', '生命科/産業生命科']

interest_columns = ['旅行','読書', '音楽', 'スポーツ', '映画・ドラマ', 'ゲーム', 'アニメ・漫画']
meta_columns = ['性別', '文理', '偏差値','満足度']
character_columns = ['ISTJ(ロジスティシャン)','ISFJ(擁護者)','INFJ(提唱者)','INTJ(建築家)','ISTP(巨匠)',
                     'ISFP(冒険家)','INFP(仲介者)','INTP(論理学者)','ESTP(起業家)','ESFP(エンターテイナー)',
                     'ENFP(運動家)','ENTP(討論者)','ESTJ(幹部)','ESFJ(領事)','ENFJ(主人公)','ENTJ(指揮官)']
subject_columns = ['国語','数学','英語','理科','社会']
pleace_columns = ['北海道','青森','岩手','宮城','秋田','山形','福島','茨城','栃木','群馬','埼玉','千葉','東京',
                  '神奈川','新潟','富山','石川','福井','山梨','長野','岐阜','静岡','愛知','三重','滋賀','京都',
                  '大阪','兵庫','奈良','和歌山','鳥取','島根','岡山','広島','山口','徳島','香川','愛媛','高知',
                  '福岡','佐賀','長崎','熊本','大分','宮崎','鹿児島','沖縄']

course_df = df[course_columns]
features_df = df[interest_columns + meta_columns + character_columns + subject_columns + pleace_columns]

def recommend_courses(user_features, course_df, features_df, top_n=3, M=4, lr=0.001, k=0.5, E=3000):
    new_features_df = pd.DataFrame([user_features], columns=features_df.columns)
    full_features_df = pd.concat([features_df, new_features_df], ignore_index=True)

    dummy_row = pd.Series([np.nan] * course_df.shape[1], index=course_df.columns)
    full_course_df = pd.concat([course_df, dummy_row.to_frame().T], ignore_index=True)

    n, D = full_course_df.shape
    U = np.random.normal(1, 0.25, (n, M))
    V = np.random.normal(1, 0.25, (D, M))
  
for _ in range(E):
    pred = np.dot(U, V.T)
    error = full_course_df.values - pred
    error[np.isnan(full_course_df.values)] = 0
    gradU = 2 * np.dot(error, V) - 2 * k * U
    gradV = 2 * np.dot(error.T, U) - 2 * k * V
    U += lr * gradU
    V += lr * gradV

    final_pred = np.dot(U, V.T)
    user_pred = pd.Series(final_pred[-1], index=course_df.columns)
    recs = user_pred.sort_values(ascending=False).head(top_n)
    return recs

st.title("京産大 進路推薦システム")
st.write("あなたの関心や特徴から、最適な学科を推薦します。")

user_features = []

st.subheader("1. 興味・関心のあるものを選んでください")
for col in interest_columns:
    val = st.checkbox(col)
    user_features.append(1 if val else 0)

st.subheader("2. あなたの属性を入力してください")
gender = st.selectbox("性別", options=["男性", "女性"])
bunri = st.selectbox("文理選択", options=["文系", "理系"])
hensachi = st.slider("現在の偏差値（目安）", 35, 70, 50)
mbti = st.selectbox("MBTI", option=["ISTJ(ロジスティシャン)","ISFJ(擁護者)","INFJ(提唱者)","INTJ(建築家)","ISTP(巨匠)",
                     "ISFP(冒険家)","INFP(仲介者)","INTP(論理学者)","ESTP(起業家)","ESFP(エンターテイナー)",
                     "ENFP(運動家)","ENTP(討論者)","ESTJ(幹部)","ESFJ(領事)","ENFJ(主人公)","ENTJ(指揮官)"])
kamoku = st.selectbox("得意科目", opyion=["国語","数学","英語","理科","社会"])
shusshin = st.selectbox("出身地", option=["北海道","青森","岩手","宮城","秋田","山形","福島","茨城","栃木","群馬","埼玉","千葉","東京",
                  "神奈川","新潟","富山","石川","福井","山梨","長野","岐阜","静岡","愛知","三重","滋賀","京都",
                  "大阪","兵庫","奈良","和歌山","鳥取","島根","岡山","広島","山口","徳島","香川","愛媛","高知",
                  "福岡","佐賀","長崎","熊本","大分","宮崎","鹿児島","沖縄"])

user_features += [0 if gender == "男性" else 1]
user_features += [0 if bunri == "文系" else 1]
user_features += [hensachi]
user_features += [mbti]
user_features += [kamoku]
user_features += [shusshin]

if st.button("進路を推薦する"):
    recs = recommend_courses(user_features, course_df, features_df, top_n=3)
    st.subheader("あなたにおすすめの学科")
    for idx, (name, score) in enumerate(recs.items(), 1):
        st.write(f"{idx}. {name}（予測スコア: {score:.2f}）")
