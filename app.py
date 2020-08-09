from flask import Flask,render_template,url_for,request,get_template_attribute,redirect,jsonify
import pandas as pd 
import numpy as np
import pickle


app = Flask(__name__)

# load the model from disk
dt=pickle.load(open('pred_data.pkl', 'rb'))
scalars=pickle.load(open('scalars.pkl', 'rb'))

dtr_model=pickle.load(open('dtr_model.pkl', 'rb'))
rtr_model=pickle.load(open('rtr_model.pkl', 'rb'))
svr_model=pickle.load(open('svr_model.pkl', 'rb'))
xgb_model=pickle.load(open('xgb_model.pkl', 'rb'))
gb_model=pickle.load(open('gradboost_model.pkl', 'rb'))
linear_model=pickle.load(open('linear_model.pkl', 'rb'))


dtr_r2=np.round(dtr_model[1:],2)
rtr_r2=np.round(rtr_model[1:],2)
svr_r2=np.round(svr_model[1:],2)
xgb_r2=np.round(xgb_model[1:],2)
gb_r2=np.round(gb_model[1:],2)
lr_r2=np.round(linear_model[1:],2)


l={'dtr':[],'rtr':[],
           'svr':[],'xgb':[],
           'gb':[],'ann':[],'lr':[]}

@app.route('/')
def home():
        l['dtr']=[]
        l['rtr']=[]
        l['svr']=[]
        l['xgb']=[]
        l['gb']=[]
        l['lr']=[]
        l['ann']=[]

        return render_template('index.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=' ',
                               prediction_dtr=[],prediction_rtr=[],prediction_svr=[],prediction_xgb=[],prediction_gb=[],prediction_lr=[],prediction_ann=[])
       
if __name__ == '__main__':
	app.run(debug=True,use_reloader=True)
