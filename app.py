from flask import Flask,render_template,url_for,request,get_template_attribute,redirect,jsonify
import pandas as pd 
import tensorflow
import numpy as np
import pickle


# load the model from disk

dt=pickle.load(open('pred_data.pkl', 'rb'))
rnn_dt=pickle.load(open('rnn_preddata.pkl', 'rb'))
scalars=pickle.load(open('scalars.pkl', 'rb'))

dtr_model=pickle.load(open('dtr_model.pkl', 'rb'))
rtr_model=pickle.load(open('rtr_model.pkl', 'rb'))
svr_model=pickle.load(open('svr_model.pkl', 'rb'))
xgb_model=pickle.load(open('xgb_model.pkl', 'rb'))
gb_model=pickle.load(open('gradboost_model.pkl', 'rb'))
linear_model=pickle.load(open('linear_model.pkl', 'rb'))

ann=tensorflow.keras.models.load_model("ann")
ann_model=pickle.load(open('ann_model.pkl', 'rb'))
rnn_model=pickle.load(open('rnn_model.pkl', 'rb'))
rnn=tensorflow.keras.models.load_model("rnn")

dtr_r2=np.round(dtr_model[1:],2)
rtr_r2=np.round(rtr_model[1:],2)
svr_r2=np.round(svr_model[1:],2)
xgb_r2=np.round(xgb_model[1:],2)
gb_r2=np.round(gb_model[1:],2)
lr_r2=np.round(linear_model[1:],2)
ann_r2=np.round(ann_model[0:],2)
rnn_r2=np.round(rnn_model[0:],2)
print(rnn_r2)
app = Flask(__name__)

l={'dtr':[],'rtr':[],
           'svr':[],'xgb':[],
           'gb':[],'ann':[],'lr':[],'rnn':[]}

@app.route('/')
def home():
        l['dtr']=[]
        l['rtr']=[]
        l['svr']=[]
        l['xgb']=[]
        l['gb']=[]
        l['lr']=[]
        l['ann']=[]
        l['rnn']=[]
        return render_template('index.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,rnn_r2=rnn_r2,
                               prediction_dtr=[],prediction_rtr=[],prediction_svr=[],prediction_xgb=[],prediction_gb=[],prediction_lr=[],prediction_ann=[],prediction_rnn=[])
                           

@app.route('/predict',methods=['GET','POST'])
def predict():
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

    y_pred_fin=[]
    x=np.array(np.reshape(dt[0][0],(1,8)))
    x=scalars[0].transform(x)
    y_pred=ann.predict(x)
    y_pred=scalars[1].inverse_transform(y_pred)
    y_pred=np.round(y_pred,3)
    y_pred_fin.append([y_pred[0],dt[2][0]])
    dt[0][1][7]=y_pred[0]
    for i in range(1,7):
        x=np.array(np.reshape(dt[0][i],(1,8)))
        x=scalars[0].transform(x)
        y_pred=ann.predict(x)
        y_pred=scalars[1].inverse_transform(y_pred)
        y_pred=np.round(y_pred,3)
        y_pred_fin.append([y_pred[0],dt[2][i]])
        dt[0][i+1][7]=y_pred[0]
    
    predic_ann=y_pred_fin
    l['ann']=predic_ann
    
    y_pred_fin=[]
    x=rnn_dt[0]
    y_pred=rnn.predict(rnn_dt[0])
    
    date=rnn_dt[2]
    y=[]
    for i in range(0,7):
        t=[]
        t.append(np.round(y_pred[i][0],3))
        t.append(date[i])
        y.append(t)
        
    #print(rnn_dt[2])
    
    predic_rnn=y
    l['rnn']=predic_rnn
   
    return render_template('predict.html',dtr_r2=dtr_r2,rtr_r2=rtr_r2,svr_r2=svr_r2,xgb_r2=xgb_r2,gb_r2=gb_r2,lr_r2=lr_r2,ann_r2=ann_r2,rnn_r2=rnn_r2,
                           prediction_dtr=l['dtr'],prediction_rtr=l['rtr'],prediction_svr=l['svr'],prediction_xgb=l['xgb'],prediction_gb=l['gb'],prediction_lr=l['lr'],prediction_ann=l['ann'],prediction_rnn=l['rnn'])

    
if __name__ == '__main__':
	app.run(debug=True,use_reloader=True)
