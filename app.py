from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from keras import layers
import keras
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def randomColor():
    r = lambda: random.randint(0,255)
    return ('#%02X%02X%02X' % (r(),r(),r()))
randomColor()

def get_variable_name(variable):
    globals_dict = globals()
    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]
def transform_input_data(X, time_steps, step=1):
    Xs = []
    
    for i in range(0, len(X) - time_steps, step):
        x = X.iloc[i:(i + time_steps)].values
        Xs.append(x)

    return np.array(Xs)
def predict_fall(filename):
    # Preprocess the ECG data
    # Feed the preprocessed data into the ML model
    # Example:
    model = tf.keras.models.load_model('notebook/mhealth_best_lstm_attn.keras')
    df = pd.read_csv(f'uploads/{filename}')
    df = df.dropna(how='any',axis=0)
    X = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    cols = X.columns
    sc = StandardScaler()
    X = pd.DataFrame(sc.fit_transform(X), columns=cols)
    X_tranformed = transform_input_data(X, 23, step=50)
    ans=model.predict(X_tranformed)
    return ans.tolist()

# @app.route('/')
# def index():
#     return render_template('home.html')

@app.route('/harsh')
def harsh():
    return render_template('harsh.html')

@app.route('/')
def upload():
    return render_template('test.html')
@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return predict_fall(filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/view-csv/<filename>')
def view_csv(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(file_path)
    return render_template('view_csv.html', data=df.to_html())

if __name__ == '__main__':
    app.run(debug=True)
