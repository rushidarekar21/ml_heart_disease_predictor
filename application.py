from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('KNNmodel1.pkl','rb'))
data = pd.read_csv('heart_disease_data.csv')

@app.route  ('/',methods=['GET','POST'])
def index():
    sex = data['sex'].unique()
    chest_pain = data['cp'].unique()
    fasting_blood_sugar = data['fbs'].unique()
    chest_pain_during_ex = data['exang'].unique()
    rest = data['restecg'].unique()
    slope = data['slope'].unique()
    ca = data['ca'].unique()
    thal = data['thal'].unique()
    return render_template('index.html', sex = sex, fasting_blood_sugar = fasting_blood_sugar,chest_pain = chest_pain,chest_pain_during_ex = chest_pain_during_ex,rest = rest,slope=slope,ca=ca,thal=thal)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    age = request.form.get('age')
    trestbps = request.form.get('trestbps')
    chol = request.form.get('chol')
    thalach = request.form.get('thalach')
    oldpeak = request.form.get('oldpeak')
    sex = request.form.get('sex')
    cp = request.form.get('chest_pain')
    fbs = request.form.get('fasting_blood_sugar')
    exang = request.form.get('chest_pain_during_ex')
    restecg = request.form.get('rest')
    slope = request.form.get('slope')
    ca = request.form.get('ca')
    thal = request.form.get('thal')

    prediction = model.predict(pd.DataFrame(
        columns=['age','trestbps','chol','thalach','oldpeak', 'sex','cp', 'fbs', 'exang', 'restecg','slope', 'ca', 'thal'],
        data=np.array(
            [age,trestbps,chol,thalach,oldpeak, sex,cp, fbs, exang, restecg,slope, ca, thal]).reshape(1, 13)))
    if prediction[0] == 1:
        return  'You have heart disease'
    else :
        return 'You are healthy'



if __name__ == '__main__':
    app.run(debug=True)




