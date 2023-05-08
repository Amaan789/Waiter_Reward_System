import pickle
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
xgb = pickle.load(open("XGB_model.pkl", 'rb'))
answer = {0:"Paltinum", 1:"Gold", 2:'Silver', 3:"Bronze"}

@app.route("/predict", methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = xgb.predict([np.array(list(data.values()))])

    return jsonify(answer[int(prediction)])

if __name__ == "__main__":
    app.run(debug=True)