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
