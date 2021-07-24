import os
from flask import Flask, request, jsonify
from machine_translator_mbart import Translator
from tensorflow import keras
import pickle


app = Flask(__name__)
translator = Translator()

@app.route('/', methods=["GET"])
def health_check():
    return "Machine Translation service is up and running"

@app.route('/supported_languages', methods=["GET"])
def get_supported_languages():
    langs = translator.get_supported_languages()
    return jsonify({"output":langs})

@app.route('/translate', methods=["GET","POST"])
def get_prediction():
    
    source = request.args.get('source')
    target = request.args.get('target')
    text = request.args.get('text')
    filename = f"cache/{source}-{target}.pkl"
    if os.path.exists(filename):
    	cache = pickle.load(open(filename,"rb"))
    else:
    	cache = dict()
    if text not in cache:
    	translation = translator.translate(text, source, target)
    	cache[text]=translation[0]
    	pickle.dump(cache,open(filename,"wb"))
    return jsonify({"output":cache[text]})

if __name__=='__main__':
    app.run(debug=True,host="0.0.0.0",port=7584)
