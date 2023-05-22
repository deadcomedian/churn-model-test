import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from flask import Flask, request
from collections import OrderedDict


app = Flask(__name__)


def classificator(data_frame):
    for feature in list(data_frame.columns):
        if data_frame[feature].dtype == 'O':
            data_frame[feature].replace(['Yes', 'No'], [1, 0], inplace=True)
            for iteration, value in enumerate(list(data_frame[feature].unique())):
                if type(value) == str:
                    if data_frame[feature].nunique() > 2:
                        iteration += 1
                    data_frame[feature].replace(value, iteration, inplace=True)


@app.route('/analyze', methods=['POST'])
def analyze():
    # read data
    data = pd.read_csv(request.files['file'])

    # prepare data
    del data['customerID']

    new_type_list = []
    for i in data['TotalCharges']:
        try:
            i = float(i)
        except:
            i = 0
        new_type_list.append(i)
    data['TotalCharges'] = new_type_list

    classificator(data)

    data["Churn"] = data["Churn"].astype(int)

    # split data to test and train
    Y = data['Churn']
    X = data.drop(labels=["Churn"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, stratify=Y, random_state=17)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # create, train & predict model
    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)
    prediction_test = model.predict(X_test)

    # examine the impact of every property from dataset
    weights = pd.Series(model.coef_[0], index=X.columns.values)
    weights = weights.sort_values(ascending=False)
    weights = weights.to_dict(OrderedDict)
    weights = sorted(weights.items(), key=lambda x: -x[1])

    # obtain accuracy
    accuracy = metrics.accuracy_score(y_test, prediction_test)

    # return results
    result = {
        'Accuracy': accuracy,
        'Weights': weights
    }

    return result


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
