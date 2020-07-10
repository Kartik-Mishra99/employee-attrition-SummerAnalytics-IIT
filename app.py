import pickle
from flask import Flask, render_template, request
import numpy as np

model = pickle.load(open('RFregression_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	overtime= int(request.form['OverTime'])
    	totalworkyears = int(request.form['TotalWorkingYears'])
    	Monthyincome= int(request.form['MonthlyIncome'])
    	age = int(request.form['Age'])
    	Yearsatcompany = int(request.form['YearsAtCompany'])
    	StockOptionlevel = int(request.form['StockOptionLevel'])
    	yearscurrentrole = int(request.form['YearsInCurrentRole'])
    	JobSatisfication = int(request.form['JobSatisfaction'])
    	maritalstatus = int(request.form['MaritalStatus'])
    	jobrole = int(request.form['JobRole'])
    	communicationskill = int(request.form['CommunicationSkill'])
    	DistanceFromHome = int(request.form['DistanceFromHome'])
    	EnvironmentSatisfaction = int(request.form['EnvironmentSatisfaction'])
    	YearsWithCurrManager = int(request.form['YearsWithCurrManager'])
    	PercentSalaryHike = int(request.form['PercentSalaryHike'])

    	data = np.array([[overtime,Monthyincome,age,totalworkyears,Yearsatcompany,JobSatisfication,yearscurrentrole,maritalstatus,jobrole,StockOptionlevel,DistanceFromHome,
    		YearsWithCurrManager,communicationskill,EnvironmentSatisfaction,PercentSalaryHike]])

    	preds = model.predict(data)
    	return render_template('result.html',prediction=preds)
    return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)



