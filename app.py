# from flask import Flask, request, render_template
# import pickle
# import numpy as np
# import xgboost as xgb
# import pandas as pd

# app = Flask(__name__)
# model = pickle.load(open('filteredwrappingXGB_XV.pkl', 'rb'))

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
#     app.run(debug=True)
#     app.run(debug=True, host="54.87.188.151", port=8000)

# from flask import Flask, request, jsonify
# import pandas as pd
# import xgboost as xgb
# from sklearn.preprocessing import LabelEncoder
# import pickle

# # Load the XGBoost model
# model_path = "filteredwrappingXGB_XV.pkl"
# with open(model_path, "rb") as f:
#     model = pickle.load(f)
# # Create a Flask app
# app = Flask(__name__)
# # Define a function to preprocess the data and make predictions


# def predict_wrapping(data):
#     # Convert categorical variables to numeric values
#     cat_cols = ["headdirection", "facebundles", "goods",
#                 "haircolor", "samplescollected", "ageatdeath"]
#     for col in cat_cols:
#         encoder = LabelEncoder()
#         data[col] = encoder.fit_transform(data[col])
#     # Make predictions using the XGBoost model
#     dmatrix = xgb.DMatrix(data)
#     predictions = model.predict(dmatrix)
#     # Convert numeric predictions to string labels
#     label_mapping = {0: "B", 1: "H", 2: "W"}
#     predicted_labels = [label_mapping[int(pred)] for pred in predictions]
#     return predicted_labels
# # Define a Flask route for handling API requests


# @app.route("/predict", methods=["POST"])
# def predict():
#     # Get the data from the request
#     data = request.get_json()
#     # Convert the data to a pandas DataFrame
#     df = pd.DataFrame.from_dict(data, orient="index").transpose()
#     # Make predictions
#     predictions = predict_wrapping(df)
#     # Return the predictions as a JSON response
#     response = {"predictions": predictions}
#     return jsonify(response)


# # Run the Flask app
# if __name__ == "__main__":
#     app.run(debug=True)


# MOST PROGRESS SO FAR -------------------------------------------------

from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.calibration import LabelEncoder
import xgboost as xgb
import pandas as pd

# Load the XGBoost model
model_path = "filteredwrappingXGB_XV.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Create a Flask app
app = Flask(__name__)

# Define a function to preprocess the data and make predictions


def predict_wrapping(data):
    # Convert categorical variables to numeric values
    cat_cols = [0, 2, 3, 4, 5, 7]  # column indices for categorical variables
    for col in cat_cols:
        encoder = LabelEncoder()
        data[:, col] = encoder.fit_transform(data[:, col])
    # Convert the data to a DMatrix object
    dmatrix = xgb.DMatrix(data)
    # Make predictions using the XGBoost model
    predictions = model.predict(dmatrix)
    # Convert numeric predictions to string labels
    label_mapping = {0: "B", 1: "H", 2: "W"}
    predicted_labels = [label_mapping[int(pred)] for pred in predictions]
    return predicted_labels

# Define a Flask route for handling API requests


@app.route("/", methods=["GET"])
def predict_api():
    input = [float(x) for x in request.form.values()]
    final_input = np.array(input).reshape(1, -1)
    prediction = model.predict(xgb.DMatrix(final_input))
    # Convert numeric predictions to string labels
    label_mapping = {0: "B", 1: "H", 2: "W"}
    predicted_label = label_mapping[int(prediction[0])]
    # Render the prediction in the prediction.html template
    return render_template('prediction.html', predicted_label=predicted_label)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)

# END MOST PROGRESS SO FAR -----------------------------------------------------------


# import pickle
# import pandas as pd
# import xgboost as xgb
# from flask import Flask, request, jsonify
# from sklearn.metrics import accuracy_score
# # Load the saved model from the pickle file
# model_path = "filteredwrappingXGB_XV.pkl"
# with open(model_path, "rb") as f:
#     model = pickle.load(f)
# # Initialize the Flask app
# app = Flask(__name__)


# @app.route("/predict", methods=["POST"])
# def predict():
#     input_data = request.json
#     # Assuming the input_data is a list of dictionaries containing the features
#     df = pd.DataFrame(input_data)
#     # Perform data preprocessing as in the original script
#     df["depth"] = df["depth"].astype(float)
#     df["length"] = df["length"].astype(float)
#     mapping = {"W": 2, "H": 1, "B": 0}
#     df["wrapping"] = df["wrapping"].map(mapping)
#     X = df.drop(columns=["wrapping"])
#     cat_cols = ["headdirection", "facebundles", "goods",
#                 "haircolor", "samplescollected", "ageatdeath"]
#     for col in cat_cols:
#         X[col] = X[col].astype("category")
#     # Create a DMatrix object for the input data
#     dtest = xgb.DMatrix(X, enable_categorical=True)
#     # Make predictions
#     y_pred = model.predict(dtest)
#     # Return the predictions as a JSON object
#     return jsonify(y_pred.tolist())


# if __name__ == "__main__":
#     app.run(debug=True)
