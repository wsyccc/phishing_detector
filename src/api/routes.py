from flask import request, jsonify, Flask
from flask_cors import CORS

from src.api.get_prediction import get_result

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    url = data["url"]
    result = get_result(url)
    return jsonify({"url": url, "prediction": int(result)}), 200

if __name__ == "__main__":
    # Run the Flask app on port 5000
    app.run(host="0.0.0.0", port=5000)