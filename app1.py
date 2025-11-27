import streamlit as st

st.title("こんにちは、吉村ゼミ")

name = st.text_input("好きな言葉を入力してください")

st.write(name)

st.checkbox("同意します")

address = st.selectbox("次の中から現住所を教えてください",["京都府","大阪府"])
st.write(address)

hobby = st.multiselect("趣味を次から複数選択してください",["映画","音楽","散歩"])
st.write(hobby)

score = st.slider("この映画を10点満点で評価してください",0,10,5)
st.write(score)

list = [
    {"latitude":35.05,"longitude":135.76},
    {"latitude":35.04,"longitide":135.75},
]
st.map(list)

st.radio("性別を選択してください",["男性","女性"])
camera = st.camera_input("写真を撮影します！")
if camera:
    st.image(camera, caption="写真", use_column_width=True)
