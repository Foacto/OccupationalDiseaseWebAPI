from DGCNN import DGCNN
from datetime import datetime
    
def get_feature_list(model):
    return model.get_feature_list()

def model_predict(model, input_data):
    # kq = model.predict(input_data)
    # if kq[0] == 1:
    #     kq = 'Mắc bệnh nghề nghiệp'
    # else:
    #     kq = 'Không mắc bệnh nghề nghiệp'
    # return kq
    return 'Không mắc bệnh nghề nghiệp'

def load_DGCNN_model():
    return DGCNN("model")

def save_message(email, name, message):
    today = datetime.utcnow().strftime('%Y-%m-%d')

    data = \
f'''email: {email},
name: {name},
message: {message};
'''

    with open(f'contact_receive/{today}.txt', 'a') as f:
        f.write(data)