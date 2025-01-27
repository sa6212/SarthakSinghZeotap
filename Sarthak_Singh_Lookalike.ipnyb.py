import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

print("Loading dataset...")
customers = pd.read_csv('Customers.csv')
print(customers.head())

print("Encoding categorical data...")
label_encoder = LabelEncoder()
customers['Region_Encoded'] = label_encoder.fit_transform(customers['Region'])

print("Processing date column...")
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['DaysSinceSignup'] = (pd.Timestamp.now() - customers['SignupDate']).dt.days

print("Selecting features for the model...")
features = customers[['Region_Encoded', 'DaysSinceSignup']]

print("Normalizing features...")
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

print("Computing cosine similarity...")
similarity_matrix = cosine_similarity(features_scaled)

customer_ids = customers['CustomerID'].values
customer_id_to_index = {id: idx for idx, id in enumerate(customer_ids)}

print("Finding top 3 lookalikes for each customer...")
lookalike_data = {}
for cust_id in customer_ids[:20]:
    idx = customer_id_to_index[cust_id]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_lookalikes = similarity_scores[1:4]
    lookalike_data[cust_id] = [(customer_ids[i], round(score, 4)) for i, score in top_lookalikes]

print("Saving results to Lookalike.csv...")
lookalike_map = pd.DataFrame({
    "cust_id": lookalike_data.keys(),
    "lookalikes": [str(value) for value in lookalike_data.values()]
})
lookalike_map.to_csv('Lookalike.csv', index=False)

print("Lookalike.csv has been successfully generated!")
