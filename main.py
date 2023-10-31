from flask import Flask, request, render_template
from utils import DGCNN
import json
import pickle
import numpy as np

app = Flask(__name__)

model = DGCNN("model")

#Web params
FEATURES_FILE_NAME = "FEATURE_INFORMATION.pkl"
open_file = open(FEATURES_FILE_NAME, "rb")
feature_list = pickle.load(open_file)
selected_feature = None


@app.route('/chooseFeature', methods=['POST'])
def chooseFeatureM():
    global selected_feature

    selected_feature_list = request.form.getlist('selectfeature')
    # print(selected_feature_list)

    if len(selected_feature_list) == 0:
        return render_template("home.html", 
        features = feature_list, message = 'Choose atleast 1 feature before submit!')

    selected_feature = []
    for feature in feature_list:
        if feature['name'] in selected_feature_list:
            feature['val'] = feature['common']
            selected_feature.append(feature)

    return render_template("home.html", 
    features = feature_list, selectedfeature = selected_feature)

@app.route('/')
def home():
    return render_template("home.html", features = feature_list, message = 'Please select desired input information!')

@app.route('/predict', methods=['POST'])
def predict():
    res = {"success": False, "result": None}
    if request.data:
        jsn = request.json
        res["success"] = True
        res["result"] = model.predict(jsn)
    return json.dumps(res, ensure_ascii=False)

@app.route('/submit', methods=['POST'])
def submit():
    # Get input data
    input_data = [{}]
    for feature in selected_feature:
        value = request.form.get(feature['name']).lower()
        feature['val'] = value
        if not value:
            value = None
        elif feature['name'] == 'vungtonthuong':
            l = value.split(',')
            if not l[0] or l[0] == ' ':
                value = None
            else:
                value = len(l)
        else:
            try:
                value = float(value)
            except ValueError as e:
                value = value
        
        if feature['name'] in ['B13','C7','C13','D2','D16','D17','E4','F5','F10','F12']:
            value = str(value)
        
        input_data[0][feature['name']] = value
    
    kq = model.predict(input_data)
    if kq[0] == 1:
        kq = 'Mắc bệnh nghề nghiệp'
    else:
        kq = 'Không mắc bệnh nghề nghiệp'
    return render_template("home.html", 
    features = feature_list, selectedfeature = selected_feature, ketqua = kq)

if __name__ == '__main__':
    print("Api run!")
    # load_DGCNN_model(model=model)
    app.run(debug=True)