# prediction function
import pickle
from flask import Flask,request,render_template
import numpy as np
app=Flask(__name__)
loaded_model = pickle.load(open("model.pkl", "rb"))
@app.route('/')
def home():
	return render_template("form.html")

def ValuePredictor(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1,-1)
	result1 = loaded_model.predict(to_predict)
	return result1[0]

@app.route('/predict', methods = ['POST'])
def predict():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())[1:]
		to_predict_list = list(map(float, to_predict_list))
		result1= ValuePredictor(to_predict_list)		
		return render_template("form.html", prediction_text= "Maximum Loan that Can be given Rs {} /-".format(result1))
if __name__=="__main__":
	app.run(debug=True)