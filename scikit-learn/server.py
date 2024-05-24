import joblib
import numpy as np

from flask import Flask
from flask import jsonify

app = Flask(__name__)

#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    x_test = np.array([7.384402835,7.247597134,1.479204416,1.481348991,0.834557652,0.611100912,0.435539722,0.287371516,2.187264442])
    prediction = model.predict(x_test.reshape(1,-1))
    return jsonify({'prediccion': list(prediction)})

if __name__ == '__main__':
    model = joblib.load('./Modelos/best_model.pkl')
    app.run(port=8080)


