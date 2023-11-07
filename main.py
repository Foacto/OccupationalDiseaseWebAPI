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
patient_features = ['id','tinh','hoten','gioitinh','namsinh','sdt','cao','can','hatd','hatt','mach',
                    'theluc','longnguc','khoangls','rungthan','rungt1','rungt2','rungt3','rungt4',
                    'rungt5','rungt6','rungg1','rungg2','rungg3','rungg4','rungg5','rungg6','go',
                    'goduc1','goduc2','goduc3','goduc4','goduc5','goduc6','riraopn','ran','ranam',
                    'ranno','ranrit','ranngay','vtam1','vtam2','vtam3','vtam4','vtam5','vtam6',
                    'vtno1','vtno2','vtno3','vtno4','vtno5','vtno6','vtrit1','vtrit2','vtrit3','vtrit4',
                    'vtrit5','vtrit6','vtngay1','vtngay2','vtngay3','vtngay4','vtngay5','vtngay6',
                    'chatluongphim','ketqua','matdotonthuong','kichthuoctt','tonthuongkhac',
                    'A6','A12','B1','B2','B3','B8','B9','B10','B11','B12','C1','C2','C3','C4','C5',
                    'C6','C8','C9','C10','C11','C12','D1','D3','D5','D7','D9','D10','D11',
                    'D12','D12_nam','D13','D13_nam','D14','D15','F6','F7','F8','F9','F11','F13','fvclt',
                    'fvctt','fev1lt','fev1tt','fvc','fev1','gaenler']
work_features = ['cviec','pxuong','tuoinghe','nampx','cv5nam','cviec1','tgian1','cviec2','tgian2',
                 'A7','A9a','A9b','A10','A11','E1','E2','E3','E5','E6','E7','F3','F4mu','F4ung',
                 'F4KhauTrang','F4gang','F4BHLD','F4Kinh']
habit_features = ['hutthuoc','slthuoc','']
medical_history_features = ['tiensuhh','benhhh','tsnoi','tsngoai','bnncuthe','nam','sobh',]
symptom_features = ['ho','tdho','tsho','n1','khacdom','loaidom','tdkhacdo','khotho','mdkhotho',
                    'tdkhotho','daunguc','vitridau','daulan','tcdau','n2','tgdau','ytodau','n3','ytotang',
                    'chaymui','khan','khokhe','dhkhac','metmoi','sutcan','socansut','tgsut',
                    '']

@app.route('/chooseFeature', methods=['POST'])
def chooseFeatureM():
    global selected_feature

    selected_feature_list = request.form.getlist('selectfeature')
    # print(selected_feature_list)
    selected_feature_list.insert(0, 'id')

    if len(selected_feature_list) == 0:
        return render_template("home.html", 
        features = feature_list,
        patient = patient_features, work = work_features, habit = habit_features, medical = medical_history_features, symptom = symptom_features,
        message = 'Choose atleast 1 feature before submit!')

    selected_feature = []
    for feature in feature_list:
        if feature['name'] in selected_feature_list:
            feature['val'] = feature['common']
            selected_feature.append(feature)

    return render_template("home.html", 
    features = feature_list,
    patient = patient_features, work = work_features, habit = habit_features, medical = medical_history_features, symptom = symptom_features,
    selectedfeature = selected_feature)

@app.route('/')
def home():
    return render_template("home.html", features = feature_list,
                           patient = patient_features, work = work_features, habit = habit_features, medical = medical_history_features, symptom = symptom_features,
                           message = 'Please select desired input information!')

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
        value = request.form.get(feature['name'])
        feature['val'] = value
        if value:
            value = value.lower()
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
    features = feature_list,
    patient = patient_features, work = work_features, habit = habit_features, medical = medical_history_features, symptom = symptom_features,
    selectedfeature = selected_feature, ketqua = kq)

if __name__ == '__main__':
    print("Api run!")
    # load_DGCNN_model(model=model)
    app.run(debug=False)