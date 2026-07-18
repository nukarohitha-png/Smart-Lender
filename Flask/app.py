from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('rdf.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))

FEATURE_ORDER = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                  'Loan_Amount_Term', 'Credit_History', 'Property_Area']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict_form():
    return render_template('predict.html')


@app.route('/output', methods=['POST'])
def output():
    form = request.form

    gender = 1 if form['gender'] == 'Male' else 0
    married = 1 if form['married'] == 'Yes' else 0
    dependents = 3 if form['dependents'] == '3+' else int(form['dependents'])
    education = 1 if form['education'] == 'Graduate' else 0
    self_employed = 1 if form['self_employed'] == 'Yes' else 0
    applicant_income = float(form['applicant_income'])
    coapplicant_income = float(form['coapplicant_income'])
    loan_amount = float(form['loan_amount'])
    loan_amount_term = float(form['loan_amount_term'])
    credit_history = float(form['credit_history'])
    property_area = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}[form['property_area']]

    features = pd.DataFrame([[gender, married, dependents, education, self_employed,
                               applicant_income, coapplicant_income, loan_amount,
                               loan_amount_term, credit_history, property_area]],
                             columns=FEATURE_ORDER)

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    confidence = round(max(model.predict_proba(features_scaled)[0]) * 100, 1)

    result = 'Approved' if prediction == 1 else 'Rejected'
    return render_template('output.html', result=result, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
