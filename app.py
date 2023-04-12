# from flask import Flask, request, render_template
# import pickle
# import numpy as np

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))


# @app.route('/')
# def home():
#     return render_template('prediction.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     input = [float(x) for x in request.form.values()]
#     final_input = [np.array(input)]
#     prediction = model.predict(final_input)

#     return render_template({'prediction': prediction.tolist()})


# if __name__ == '__main__':
#     app.run(debug=True)
#     app.run(host="0.0.0.0", port=8000)

from flask import Flask, request, render_template
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

# app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

# label_map = {0: "B", 1: "H", 2: "W"}


# @app.route('/')
# def home():
#     return render_template('prediction.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     input = [float(x) for x in request.form.values()]
#     final_input = [np.array(input)]
#     prediction = model.predict(final_input)

#     # Get the predicted index
#     predicted_index = np.argmax(prediction)

#     # Map the index to the corresponding label
#     predicted_label = label_map[predicted_index]

#     return render_template('prediction.html', output=predicted_label)


# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=8000)

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

label_map = {0: "B", 1: "H", 2: "W"}


@app.route('/')
def home():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    input = [float(x) for x in request.form.values()]
    final_input = [np.array(input)]
    prediction = model.predict(final_input)

    # Get the predicted index
    predicted_index = np.argmax(prediction)

    # Map the index to the corresponding label
    predicted_label = label_map[predicted_index]

    return render_template('prediction.html', output=predicted_label)


if __name__ == '__main__':
    app.run(debug=True)
