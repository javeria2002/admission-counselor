from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load university data
df = pd.read_csv('../data/universities.csv')

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/')
def home():
    return jsonify({"message": "Admission Counselor Backend is Running! ✅"})

@app.route('/universities', methods=['GET'])
def get_universities():
    return jsonify(df.to_dict(orient='records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    merit = float(data.get('merit', 0))
    budget = float(data.get('budget', 0))
    city = data.get('city', '').lower()

    filtered = df[
        (df['min_merit'] <= merit) &
        (df['fee_per_year'] <= budget)
    ]

    if city:
        filtered = filtered[filtered['city'].str.lower() == city]

    if filtered.empty:
        return jsonify({"message": "No universities found matching your criteria."})

    return jsonify(filtered.to_dict(orient='records'))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    # Convert university data to text for AI context
    uni_data = df.to_string(index=False)

    prompt = f"""
You are a helpful university admission counselor for Pakistani students.
You have access to this university database:

{uni_data}

Answer the student's question based on this data only.
Be friendly, helpful and concise.
Always respond in English.
If asked about merit, fees, programs or cities, use the data above.

Student's question: {user_message}
"""

    response = model.generate_content(prompt)
    return jsonify({"reply": response.text})

if __name__ == '__main__':
    app.run(debug=True)