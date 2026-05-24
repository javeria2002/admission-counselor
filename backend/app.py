from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from dotenv import load_dotenv
import os
import requests as req

load_dotenv()

app = Flask(__name__)
CORS(app)

df = pd.read_csv('../data/universities.csv')

@app.route('/')
def home():
    return jsonify({"message": "Admission Counselor Backend is Running!"})

@app.route('/universities', methods=['GET'])
def get_universities():
    return jsonify(df.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    merit = float(data.get('merit', 0))
    budget = float(data.get('budget', 0))
    city = data.get('city', '').lower()
    program = data.get('program', '')

    filtered = df[
        (df['min_merit'] <= merit) &
        (df['fee_per_year'] <= budget)
    ]

    if city:
        filtered = filtered[filtered['city'].str.lower() == city]

    if program:
        filtered = filtered[filtered['programs'].str.contains(program, case=False)]

    if filtered.empty:
        return jsonify({"message": "No universities found matching your criteria."})

    return jsonify(filtered.to_dict(orient='records'))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    msg = data.get('message', '')

    uni_data = df.to_string(index=False)

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