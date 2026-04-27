import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -------------------------------
# STEP 1: Load dataset
# -------------------------------
data = pd.read_csv('groceries_dataset.csv')

# -------------------------------
# STEP 2: Create Transaction ID
# (Combine Member + Date)
# -------------------------------
data['Transaction'] = data['Member_number'].astype(str) + '_' + data['Date']

# -------------------------------
# STEP 3: Group items into baskets
# -------------------------------
transactions = data.groupby('Transaction')['itemDescription'].apply(list).values.tolist()

print("\n✅ Correct Transactions (first 5):\n")
print(transactions[:5])

# -------------------------------
# STEP 4: One-Hot Encoding
# -------------------------------
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_data, columns=te.columns_)

print("\n✅ One-Hot Encoded Data:\n")
print(df.head())

# -------------------------------
# STEP 5: Apply Apriori Algorithm
# -------------------------------
frequent_items = apriori(df, min_support=0.002, use_colnames=True)

print("\n✅ Frequent Itemsets:\n")
print(frequent_items.head())

# -------------------------------
# STEP 6: Generate Association Rules
# -------------------------------
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.1)
rules=rules[rules['lift']>1]
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

rules = rules.sort_values(by='lift', ascending=False)

print("\n✅ Association Rules:\n")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])