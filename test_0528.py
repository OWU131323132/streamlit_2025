import streamlit as st

st.title("サークル入会申し込みフォーム")

# フォームの作成
with st.form("circle_application_form"):
    # 基本情報の入力欄
    name = st.text_input("お名前")
    grade = st.selectbox("学年", ["1年生", "2年生", "3年生", "4年生"])
    activity = st.selectbox("好きな活動", ["文化祭", "合宿", "勉強会", "交流会"])
    motivation = st.text_area("意気込み")

    # 申し込むボタン
    submitted = st.form_submit_button("申し込む")

# 送信ボタンが押されたら、入力された情報を表示
if submitted:
    st.write("### 入力された情報:")
    st.write(f"**お名前:** {name}")
    st.write(f"**学年:** {grade}")
    st.write(f"**好きな活動:** {activity}")
    st.write(f"**意気込み:** {motivation}")
