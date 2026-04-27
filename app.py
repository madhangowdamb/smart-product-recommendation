import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Smart Product Recommendation", layout="wide")

# -------------------------------
# PREMIUM CSS
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    background: -webkit-linear-gradient(#38bdf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-card {
    text-align:center;
    padding:15px;
    border-radius:10px;
    background: rgba(99,102,241,0.2);
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown('<div class="big-title">🛒 Smart Product Recommendation</div>', unsafe_allow_html=True)
st.markdown("### 📊 Market Basket Analysis using Apriori")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Groceries_dataset.csv")

df = load_data()

# -------------------------------
# PREPROCESS
# -------------------------------
transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()

te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# -------------------------------
# APRIORI
# -------------------------------
frequent_items = apriori(df_encoded, min_support=0.005, use_colnames=True)

if frequent_items.empty:
    st.error("No frequent items found.")
    st.stop()

rules = association_rules(frequent_items, metric="confidence", min_threshold=0.05)

if rules.empty:
    st.warning("No rules generated.")
    st.stop()

# -------------------------------
# CLEAN RULES
# -------------------------------
def convert(x):
    return ', '.join(list(x))

rules['antecedents'] = rules['antecedents'].apply(convert)
rules['consequents'] = rules['consequents'].apply(convert)

rules = rules[(rules['lift'] > 1) & (rules['confidence'] > 0.3)]

# -------------------------------
# TOP PRODUCTS
# -------------------------------
st.markdown("## 🔥 Top Selling Products")

top_items = frequent_items.sort_values(by="support", ascending=False).head(5)

for _, row in top_items.iterrows():
    item = list(row['itemsets'])[0]
    st.markdown(f'<div class="card">🛍️ {item} <br> <small>Support: {round(row["support"],3)}</small></div>', unsafe_allow_html=True)

# -------------------------------
# SELECT PRODUCT
# -------------------------------
st.markdown("## 🔍 Select Product")

product = st.selectbox("Choose a product", sorted(df_encoded.columns))

# -------------------------------
# FILTER RULES (UNCHANGED)
# -------------------------------
result = rules[rules['antecedents'].str.startswith(product)].copy()

def simplify(x, product):
    items = x.split(', ')
    if product in items:
        items.remove(product)
    return product

result['antecedents'] = result['antecedents'].apply(lambda x: simplify(x, product))

result = result.sort_values(by=['lift', 'confidence'], ascending=False)

result['confidence'] = result['confidence'].round(2)
result['lift'] = result['lift'].round(2)

result = result[['antecedents', 'consequents', 'confidence', 'lift']]

# LIMIT
result = result.head(10)

# -------------------------------
# DISPLAY RECOMMENDATIONS
# -------------------------------
st.markdown("## 📊 Recommendations")

if result.empty:
    st.warning("No recommendations found.")
else:
    # METRICS
    col1, col2, col3 = st.columns(3)

    col1.markdown(f'<div class="metric-card">Confidence<br><b>{result.iloc[0]["confidence"]}</b></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="metric-card">Lift<br><b>{result.iloc[0]["lift"]}</b></div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="metric-card">Rules<br><b>{len(result)}</b></div>', unsafe_allow_html=True)

    st.markdown("---")

    # CARD STYLE RECOMMENDATIONS
    for i, row in result.iterrows():
        st.markdown(
            f"""
            <div class="card">
            🛒 If customer buys <b>{product}</b><br>
            👉 Recommended: <b>{row['consequents']}</b><br>
            <small>Confidence: {row['confidence']} | Lift: {row['lift']}</small>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("💡 Lift > 1 = strong association")