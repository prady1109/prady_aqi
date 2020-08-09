from flask import Flask,render_template,url_for,request,get_template_attribute,redirect,jsonify
import pandas as pd 

import numpy as np
import pickle


app = Flask(__name__)

dtr_model=pickle.load(open('dtr_model.pkl', 'rb'))
dtr_r2=np.round(dtr_model[1:],2)
@app.route('/')
def home():
        return render_template('ind.html',nam=dtr_r2)
                           


if __name__ == '__main__':
	app.run(debug=True,use_reloader=True)
