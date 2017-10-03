import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods =['POST'])

def apicall():
    """ API call
    Pandas dataframe(sent as payload) from API call"""

    try:
        test_json = request.get_json()
        test = pd.read_json(test_json, orient = 'records')

        test['Dependents'] = [str(x) for x in list(test['Dependents'])]

        # getting the Loan_id seperated out
        loan_ids = test['Loan_id']

    except Exception as e:
        raise e

    clf = 'model_v1.pk'


    if test.empty:
        return(bad_request())

    else:

        # Load the saved model
        print("Loading the model...")
        loaded_model = None
        with open('./models/'+clf, 'rb') as f:
            loaded_model = pickle.laod(f)

        print("The model has been loaded... ready to prediction now...")
        predictions = loaded_model.predict(test)

        """ Add the prediction as a Series to a new pandas dataframe"""

        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

        """ We can be  as creative in sending the response.
        but we need to sedn the response codes as well """

        repsonses  = jsonify(predictions = final_predictions.to_json(orient = "records"))

        responses.status_code  = 200

        return(responses)

import json
import requests

"""Setting the headers to send and accept json responses
"""
header = {'Content-Type': 'application/json', \
                  'Accept': 'application/json'}

"""Reading test batch
"""
df = pd.read_csv('data/test.csv', encoding="utf-8-sig")
df = df.head()

"""Converting Pandas Dataframe to json
"""
data = df.to_json(orient='records')

# print data

"""POST <url>/predict
"""
resp = requests.post("http://0.0.0.0:8000/predict", \
                    data = json.dumps(data),\
                    headers= header)

resp.status_code
