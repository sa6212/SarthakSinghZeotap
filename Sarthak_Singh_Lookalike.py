import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

customers = pd.read_csv('Customers.csv')

label_encoder = LabelEncoder()
customers['Region_Encoded'] = label_encoder.fit_transform(customers['Region'])

customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
customers['DaysSinceSignup'] = (pd.Timestamp.now() - customers['SignupDate']).dt.days

features = customers[['Region_Encoded', 'DaysSinceSignup']]

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

similarity_matrix = cosine_similarity(features_scaled)

customer_ids = customers['CustomerID'].values
customer_id_to_index = {id: idx for idx, id in enumerate(customer_ids)}

lookalike_data = {}
for cust_id in customer_ids[:20]:
    idx = customer_id_to_index[cust_id]

    similarity_scores = list(enumerate(similarity_matrix[idx]))
  
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_lookalikes = similarity_scores[1:4]  
    
    lookalike_data[cust_id] = [(customer_ids[i], round(score, 4)) for i, score in top_lookalikes]


lookalike_map = pd.DataFrame({
    "cust_id": lookalike_data.keys(),
    "lookalikes": [str(value) for value in lookalike_data.values()]
})
lookalike_map.to_csv('Lookalike.csv', index=False)

print("Lookalike.csv has been successfully generated!")
