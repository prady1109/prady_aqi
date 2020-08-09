from flask import Flask,render_template,url_for,request,get_template_attribute,redirect,jsonify
import pandas as pd 

import numpy as np
import pickle


# load the model from disk
dt=pickle.load(open('pred_data.pkl', 'rb'))
scalars=pickle.load(open('scalars.pkl', 'rb'))

dtr_model=pickle.load(open('dtr_model.pkl', 'rb'))
rtr_model=pickle.load(open('rtr_model.pkl', 'rb'))
svr_model=pickle.load(open('svr_model.pkl', 'rb'))
xgb_model=pickle.load(open('xgb_model.pkl', 'rb'))
gb_model=pickle.load(open('gradboost_model.pkl', 'rb'))
linear_model=pickle.load(open('linear_model.pkl', 'rb'))

ann_model=pickle.load(open('ann_model.pkl', 'rb'))

dtr_r2=np.round(dtr_model[1:],2)
rtr_r2=np.round(rtr_model[1:],2)
svr_r2=np.round(svr_model[1:],2)
xgb_r2=np.round(xgb_model[1:],2)
gb_r2=np.round(gb_model[1:],2)
lr_r2=np.round(linear_model[1:],2)
ann_r2=np.round(ann_model[1:],2)

app = Flask(__name__)

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

        return render_template('index.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                               prediction_dtr=[],prediction_rtr=[],prediction_svr=[],prediction_xgb=[],prediction_gb=[],prediction_lr=[],prediction_ann=[])
                           

@app.route('/dtr',methods=['POST'])
def predict_dtr():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=dtr_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=dtr_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    
    predic_dtr=y_pred_fin
 
    l['dtr']=predic_dtr
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])

@app.route('/rtr',methods=['POST'])
def predict_rtr():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=rtr_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=rtr_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    #y_pred_fin = scalars[1].inverse_transform(y_pred_fin)
    predic_rtr=y_pred_fin
    l['rtr']=predic_rtr
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])


@app.route('/svr',methods=['POST'])
def predict_svr():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=svr_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=svr_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    predic_svr=y_pred_fin
    l['svr']=predic_svr
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])


@app.route('/xgb',methods=['POST'])
def predict_xgb():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=xgb_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=xgb_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    predic_xgb=y_pred_fin
    l['xgb']=predic_xgb
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])


@app.route('/gb',methods=['POST'])
def predict_gb():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=gb_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=gb_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    predic_gb=y_pred_fin
    l['gb']=predic_gb
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])


@app.route('/linear',methods=['POST'])
def predict_lr():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=linear_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=linear_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    predic_linear=y_pred_fin
    l['lr']=predic_linear
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])


@app.route('/ann',methods=['POST'])
def predict_ann():
    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=ann_model[0].predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=ann_model[0].predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    predic_ann=y_pred_fin
    l['ann']=predic_ann
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'])


if __name__ == '__main__':
	app.run(debug=True,use_reloader=True)
