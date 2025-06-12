import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="食材からレシピを探すアプリ", layout="centered")

st.title("🥘 食材からレシピを探すアプリ")

st.markdown("""
### 使い方
冷蔵庫にある食材を選ぶだけで、それを使ったレシピを提案します🍳  
複数の食材を選べば、それらをすべて使ったレシピが表示されます。  
完全に一致しない場合も、似ているレシピをおすすめします！

**例えば：**
- 「卵」「玉ねぎ」「鶏肉」 → 親子丼
- 「キャベツ」「豚肉」 → ロールキャベツ
""")

st.image("https://www.sirogohan.com/_files/recipe/images/oyakodon/oyakodon1303.JPG", 
         caption="親子丼：鶏肉と卵で作れる人気メニュー", width=500)

ingredients = [
    "卵", "鶏肉", "豚肉", "牛肉", "玉ねぎ", "にんじん", "じゃがいも",
    "キャベツ", "ピーマン", "豆腐", "ご飯", "チーズ", "トマト", "小松菜", "パン", "スパゲッティ", "ウィンナー"
]

selected_ingredients = st.multiselect("使いたい食材を選んでください（1つでもOK）", ingredients)

meal_type = st.multiselect(
    "作りたい料理のタイプを選んでください（任意）",
    ["軽食","昼食", "夕食","お弁当", "デザート"]
)

meal_purpose = st.multiselect(
    "料理の目的を選んでください（任意）",
    ["ヘルシー", "子供向け", "簡単", "がっつり"]
)

recipes = [
{
    "name": "親子丼",
    "ingredients": ["卵", "鶏肉", "玉ねぎ", "ご飯"],
    "image": "https://www.sirogohan.com/_files/recipe/images/oyakodon/oyakodon1303.JPG",
    "meal_type": ["昼食"],
    "purpose": ["子供向け", "がっつり"]
},
{
    "name": "野菜炒め",
    "ingredients": ["キャベツ", "にんじん", "ピーマン", "豚肉"],
    "image": "https://www.sirogohan.com/_files/recipe/images/yasaiitame/yasaiitameyoko.JPG",
    "meal_type": ["昼食", "夕食","お弁当"],
    "purpose": ["簡単", "子供向け"]
},
{
    "name": "肉じゃが",
    "ingredients": ["じゃがいも", "玉ねぎ", "にんじん", "牛肉"],
    "image": "https://www.sirogohan.com/_files/recipe/images/nikujaga/nikujaga1428.JPG",
    "meal_type": ["夕食"],
    "purpose": ["がっつり", "子供向け"]
},
{
    "name": "卵焼き",
    "ingredients": ["卵"],
    "image": "https://www.sirogohan.com/_files/recipe/images/atuyaki/atuyaki5472.JPG",
    "meal_type": ["朝食", "お弁当"],
    "purpose": ["簡単", "子供向け"]
},
{
    "name": "トマトチーズのカプレーゼ",
    "ingredients": ["トマト", "チーズ"],
    "image": "https://www.sirogohan.com/_files/recipe/images/soramame/kapure23772.JPG",
    "meal_type": ["夕食"],
    "purpose": ["ヘルシー","簡単"]
},
{
    "name": "ロールキャベツ",
    "ingredients": ["豚肉", "キャベツ"],
    "image": "https://www.sirogohan.com/_files/recipe/images/ro-rukyabetu/ro-rukyabetu9990.JPG",
    "meal_type": ["夕食"],
    "purpose": ["ヘルシー"]
},
{
    "name": "ピーマンの肉詰め",
    "ingredients": ["ピーマン", "豚肉", "玉ねぎ"],
    "image": "https://www.sirogohan.com/_files/recipe/images/pi-mannikudume/pi-mannikudume4666.JPG",
    "meal_type": ["夕食", "お弁当"],
    "purpose": ["子供向け", "がっつり"]
},
{
    "name": "コロッケ",
    "ingredients": ["じゃがいも", "卵", "豚肉", "牛肉"],
    "image": "https://www.sirogohan.com/_files/recipe/images/korokke/korokke6215.JPG",
    "meal_type": ["昼食", "夕食"],
    "purpose": ["子供向け", "がっつり"]
},
{
    "name": "鶏団子と春雨のスープ",
    "ingredients": ["小松菜", "鶏肉", "卵", "玉ねぎ", "にんじん"],
    "image": "https://www.sirogohan.com/_files/recipe/images/toridangosiru/toridangoharusame3487.JPG",
    "meal_type": ["朝食", "昼食"],
    "purpose": ["ヘルシー", "簡単"]
},
{
    "name": "肉豆腐",
    "ingredients": ["豆腐", "牛肉", "玉ねぎ"],
    "image": "https://www.sirogohan.com/_files/recipe/images/nikudouhu/nikudouhubig6717.JPG",
    "meal_type": ["夕食"],
    "purpose": ["がっつり"]
},
{
    "name": "にんじんサラダ",
    "ingredients": ["にんじん"],
    "image": "https://www.sirogohan.com/_files/recipe/images/ninjins/ninjins9978.JPG",
    "meal_type": ["お弁当", "昼食"],
    "purpose": ["ヘルシー"]
},
{
    "name": "ポテトサラダ",
    "ingredients": ["じゃがいも", "卵"],
    "image": "https://www.sirogohan.com/_files/recipe/images/potates/potates0500.JPG",
    "meal_type": ["お弁当", "夕食"],
    "purpose": ["子供向け", "ヘルシー"]
},
{
    "name": "グラタン",
    "ingredients": ["鶏肉", "チーズ"],
    "image": "https://www.sirogohan.com/_files/recipe/images/makaronigura/makaronigura4007.JPG",
    "meal_type": ["お弁当", "昼食"],
    "purpose": ["子供向け", "がっつり"]
},
{
    "name": "ナポリタン",
    "ingredients": ["玉ねぎ", "ピーマン", "ウィンナー"],
    "image": "https://www.sirogohan.com/_files/recipe/images/naporitan/naporitan82192.JPG",
    "meal_type": ["昼食", "夕食"],
    "purpose": ["子供向け", "がっつり"]
},
{
    "name": "オムライス",
    "ingredients": ["ご飯", "卵", "鶏肉"],
    "image": "https://www.sirogohan.com/_files/recipe/images/omuraisu/omuraisu9142.JPG",
    "meal_type": ["昼食"],
    "purpose": ["簡単", "子供向け"]
},
{
    "name": "ピザトースト",
    "ingredients": ["パン", "ピーマン", "ウィンナー"],
    "image": "https://www.sirogohan.com/_files/recipe/images/piza/pizat1711.JPG",
    "meal_type": ["朝食"],
    "purpose": ["簡単", "子供向け"]
},
{
    "name": "卵サンド",
    "ingredients": ["卵", "パン"],
    "image": "https://www.sirogohan.com/_files/recipe/images/pan/tamagosand3499.JPG",
    "meal_type": ["朝食", "お弁当"],
    "purpose": ["簡単", "子供向け"]
},
{
    "name": "玉ねぎのスープ",
    "ingredients": ["玉ねぎ"],
    "image": "https://www.sirogohan.com/_files/recipe/images/sintamakatuo/sintamakatuo6627.JPG",
    "meal_type": ["朝食", "昼食"],
    "purpose": ["簡単", "ヘルシー"]
},
{
    "name": "ドライカレー",
    "ingredients": ["ご飯", "豚肉", "牛肉", "玉ねぎ", "ピーマン", "にんじん"],
    "image": "https://www.sirogohan.com/_files/recipe/images/curry/drycurry2166.JPG",
    "meal_type": ["夕食"],
    "purpose": ["がっつり"]
},
{
    "name": "あんバタートースト",
    "ingredients": ["パン"],
    "image": "https://www.sirogohan.com/_files/recipe/images/toast/anbata0523.JPG",
    "meal_type": ["デザート", "朝食"],
    "purpose": ["簡単", "子供向け"]
},
{
    "name": "プリン",
    "ingredients": ["卵"],
    "image": "https://www.sirogohan.com/_files/recipe/images/purin/purinyoko.jpg",
    "meal_type": ["デザート"],
    "purpose": ["子供向け"],
},
{
    "name": "しじみパスタ",
    "ingredients": ["スパゲッティ", "しじみ", "ねぎ", "にんにく"],
    "image": "https://www.sirogohan.com/_files/recipe/images/sijimi/sijimip7272.JPG",
    "meal_type": ["昼食", "夕食"],
    "purpose": ["簡単"]
},
{
    "name": "和風たらこパスタ",
    "ingredients": ["スパゲッティ", "たらこ", "大葉"],
    "image": "https://www.sirogohan.com/_files/recipe/images/pasuta-mentai/pasuta-mentai0531.JPG",
    "meal_type": ["昼食", "夕食"],
    "purpose": ["簡単"]
}

]
if st.button("レシピを検索！"):
    matched = []
    for recipe in recipes:
        if all(item in recipe["ingredients"] for item in selected_ingredients) and \
           (meal_type == [] or any(mt in recipe["meal_type"] for mt in meal_type)) and \
           (meal_purpose == [] or any(mp in recipe["purpose"] for mp in meal_purpose)):
            matched.append(recipe)

    if matched:
        for recipe in matched:
            st.subheader(f"{recipe['name']}")
            st.image(recipe["image"], width=400)
            st.markdown(f"**材料**: {', '.join(recipe['ingredients'])}")
    else:
        st.warning("😢 該当するレシピが見つかりませんでした。")

    if selected_ingredients:
        selected_text = " ".join(selected_ingredients)
        recipe_texts = [" ".join(recipe["ingredients"]) for recipe in recipes]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([selected_text] + recipe_texts)
        cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

        matched_names = [r["name"] for r in matched]

        similar_results = sorted(
         [(recipes[i], score) for i, score in enumerate(cosine_sim)
          if recipes[i]["name"] not in matched_names],
         key=lambda x: x[1],
         reverse=True
       )[:3]

        if similar_results:
            st.markdown("---")
            st.subheader("👀 似ているレシピ")
            for recipe, _ in similar_results:
                st.markdown(f"### {recipe['name']}")
                st.image(recipe["image"], width=400)
                st.markdown(f"**材料**: {', '.join(recipe['ingredients'])}")
