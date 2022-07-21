from flask import Flask,request,jsonify,render_template,url_for
import pickle
import numpy as np




app=Flask(__name__)

model=pickle.load(open('alg1.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data=[list(data.values())]
    print('new_data:',new_data)
    output=model.predict(new_data)[0]
    return jsonify(output)

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    
    output=model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text="The Sales is  {}".format(output))



if __name__=='__main__':
    app.run(debug=True)