from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import pickle 

app = Flask(__name__)

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    nm = request.form['nm']

    # Load the model
    model = pickle.load(open('flask/models/stock_model.pkl', 'rb'))

    # Process the input data
    input_data = np.array([nm])  # Convert to numerical format

    # Make a prediction
    prediction = model.predict(input_data)

    return render_template('results.html', quote=nm, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)