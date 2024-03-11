from flask import Flask, request, jsonify
from score import *
import pickle
app = Flask(__name__)
loaded_model = pickle.load(open(r"D:/cmi/sem 4/AppliedML/assi3/finetunedlogistic.pkl", 'rb'))
@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text')
    if text:
        prediction, propensity = score(text, loaded_model, threshold=0.5)
        return jsonify({'prediction': prediction, 'propensity': propensity})
    else:
        return jsonify({'error': 'No text'}), 400

if __name__ == '__main__':
    app.run(debug=True,port=5000)