from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import config
from utils import CarPriceModel

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        # Get data from the request
        data = request.form.to_dict()
        df = pd.DataFrame(data, index=[0])

        # Make prediction using the model
        car_price_model = CarPriceModel()
        predicted_price = car_price_model.predict_price(df)

        # Return the result
        return render_template('index.html', predicted_price = f"${np.around(predicted_price, 3)}")

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = config.PORT_NUMBER, debug=False)

