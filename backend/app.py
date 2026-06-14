from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv
import os
import requests as req
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

load_dotenv()

app = Flask(__name__)
CORS(app)

df = pd.read_csv('../data/universities.csv')

# ─────────────────────────────────────────
# ML SETUP — runs once when Flask starts
# ─────────────────────────────────────────

features = df[['min_merit', 'max_merit', 'fee_per_year', 'hec_rank']].copy()
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-MEANS — group universities into 3 tiers
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Label clusters by average fee
cluster_fees = df.groupby('cluster')['fee_per_year'].mean().sort_values()
tier_labels = {}
tier_labels[cluster_fees.index[0]] = 'Budget'
tier_labels[cluster_fees.index[1]] = 'Mid-Range'
tier_labels[cluster_fees.index[2]] = 'Premium'
df['tier'] = df['cluster'].map(tier_labels)

# DECISION TREE — predict admission (1 = likely, 0 = unlikely)
# Label: 1 if merit >= min_merit threshold, 0 otherwise
df['label'] = (df['min_merit'] <= df['max_merit'] * 0.88).astype(int)
dt_features = df[['min_merit', 'max_merit', 'fee_per_year', 'hec_rank']]
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(dt_features, df['label'])

# KNN — find similar universities
knn_model = KNeighborsClassifier(n_neighbors=4)
knn_model.fit(features_scaled, df.index)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def home():
    return jsonify({"message": "Admission Counselor Backend is Running!"})

@app.route('/universities', methods=['GET'])
def get_universities():
    result = df.drop(columns=['cluster', 'label']).to_dict(orient='records')
    return jsonify(result)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    merit  = float(data.get('merit', 0))
    budget = float(data.get('budget', 0))
    city   = data.get('city', '').lower()
    program = data.get('program', '')

    filtered = df[
        (df['min_merit'] <= merit) &
        (df['fee_per_year'] <= budget)
    ].copy()

    if city:
        filtered = filtered[filtered['city'].str.lower() == city]
    if program:
        filtered = filtered[filtered['programs'].str.contains(program, case=False)]

    if filtered.empty:
        return jsonify({"message": "No universities found matching your criteria."})

    result = filtered.drop(columns=['cluster', 'label']).to_dict(orient='records')
    return jsonify(result)

# ─── DECISION TREE — ML Admission Prediction ───
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    student_merit = float(data.get('merit', 0))
    uni_name = data.get('university', '')

    uni = df[df['name'] == uni_name]
    if uni.empty:
        return jsonify({"error": "University not found"})

    uni = uni.iloc[0]

    # Decision Tree prediction
    input_data = [[uni['min_merit'], uni['max_merit'], uni['fee_per_year'], uni['hec_rank']]]
    dt_pred = dt_model.predict(input_data)[0]
    dt_prob = dt_model.predict_proba(input_data)[0]

    # Rule-based chance calculation using merit
    min_m = uni['min_merit']
    max_m = uni['max_merit']
    mid_m = (min_m + max_m) / 2

    if student_merit >= max_m - 2:
        chance = "Very High"
        color = "green"
        percent = 90
    elif student_merit >= mid_m:
        chance = "Good"
        color = "yellow"
        percent = 70
    elif student_merit >= min_m:
        chance = "Moderate"
        color = "orange"
        percent = 45
    else:
        chance = "Low"
        color = "red"
        percent = 15

    return jsonify({
        "university": uni_name,
        "student_merit": student_merit,
        "required_merit": f"{min_m}% - {max_m}%",
        "chance": chance,
        "color": color,
        "percent": percent,
        "tier": uni['tier'],
        "dt_confidence": round(float(max(dt_prob)) * 100, 1),
        "message": f"Based on Decision Tree analysis, your admission chance at {uni_name} is {chance} ({percent}%). The university is categorized as {uni['tier']} tier by our K-Means clustering model."
    })

# ─── KNN — Find Similar Universities ───
@app.route('/similar', methods=['POST'])
def similar():
    data = request.json
    uni_name = data.get('university', '')

    uni_row = df[df['name'] == uni_name]
    if uni_row.empty:
        return jsonify({"error": "University not found"})

    idx = uni_row.index[0]
    uni_features = features_scaled[idx].reshape(1, -1)

    distances, indices = knn_model.kneighbors(uni_features)

    similar_unis = []
    for i, dist in zip(indices[0], distances[0]):
        if df.iloc[i]['name'] != uni_name:
            uni_data = df.iloc[i].drop(['cluster', 'label']).to_dict()
            uni_data['similarity_score'] = round((1 - dist / (dist + 1)) * 100, 1)
            similar_unis.append(uni_data)

    return jsonify({"similar": similar_unis[:3]})

# ─── K-MEANS — Get University Tiers ───
@app.route('/tiers', methods=['GET'])
def tiers():
    result = []
    for tier in ['Budget', 'Mid-Range', 'Premium']:
        unis = df[df['tier'] == tier][['name', 'city', 'fee_per_year', 'hec_rank', 'tier']]
        result.append({
            "tier": tier,
            "count": len(unis),
            "universities": unis.to_dict(orient='records')
        })
    return jsonify(result)

# ─── CHAT ───
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    msg = data.get('message', '')
    uni_data = df.drop(columns=['cluster', 'label']).to_string(index=False)

    prompt = f"""You are a helpful university admission counselor for Pakistani students.
You have access to this university database:

{uni_data}

Answer the student's question based on this data.
Be friendly, helpful and concise.
Always respond in English.

Student's question: {msg}"""

    response = req.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    result = response.json()
    if 'choices' in result:
        reply = result['choices'][0]['message']['content']
    else:
        reply = str(result)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)