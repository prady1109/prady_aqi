from flask import Flask,render_template,url_for,request,get_template_attribute,redirect,jsonify
import pandas as pd 

import numpy as np
import pickle


app = Flask(__name__)


@app.route('/')
def home():
        return render_template('ind.html',nam='hi')
                           


if __name__ == '__main__':
	app.run(debug=True,use_reloader=True)
