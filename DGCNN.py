import pickle
from keras.models import load_model
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import gc

MODEL_NAME = 'model_20231007_ver02_0.7'
MID_LIST_FILE_NAME = "MID_LIST.pkl"
COLUMNS_NAME_FILE_NAME = "COLUMNS_NAME.pkl"
MAX_VALUES_FILE_NAME = "MAX_VALUES.pkl"

class DGCNN():
    model = None
    df = None
    COLUMNS_NAME = None
    MID_LIST = None
    original_df = None
    not_encoded_df = None
    max_values = None
    """docstring for DGCNN."""
    def __init__(self, name):
        if self.model is None:
            self.model = load_model(MODEL_NAME)
        if self.MID_LIST == None:
            open_file = open(MID_LIST_FILE_NAME, "rb")
            self.MID_LIST = pickle.load(open_file)
            open_file.close()
        if self.max_values is None:
            open_file = open(MAX_VALUES_FILE_NAME, "rb")
            self.max_values = pickle.load(open_file)
            open_file.close()
        self.df = pd.read_excel("data.xlsx")
        self.not_encoded_df = pd.read_excel("Main_data_NotEncoded.xlsx")
        super(DGCNN, self).__init__()
    
    def predict(self, data):
        self.df.drop(self.df.index, inplace=True)
        
        # load data to dataframe then normalize + create graph list
        for block in data:
            new_row = {}
            for col in self.df.columns:
                if col in block.keys():
                    if block[col] == None:
                        new_row[col] = np.nan
                    else:
                        new_row[col] = block[col]
                else:
                    new_row[col] = np.nan
            df_dictionary = pd.DataFrame([new_row])
            self.df = pd.concat([self.df, df_dictionary], ignore_index=True)

        #Pre-processing data
        self.encoded(self.df)
        self.normalize(self.df)

        #Create graph
        graph_X = self.create_graph_list(self.df)
        X_generator = PaddedGraphGenerator(graphs=graph_X)

        pre_gen = X_generator.flow(
            self.df.index,
            symmetric_normalization=False,
            batch_size=1,
        )
        gc.collect()
        return [round(i[0]) for i in self.model.predict(pre_gen)]
    
    def encoded(self, X):
        #Encode data by hand
        X.replace(['khong','co','ko','khong hut thuoc','co hut thuoc'], [0,1,0,0,1], inplace = True)

        X['B13'].replace(['1.0','4.0','145.0','12345.0','1234.0','134.0','1245.0','13.0','14.0',
        '1345.0','3.0','34.0','124.0','45.0','5.0','15.0','12.0','24.0','2345.0',
        '345.0','123.0','245.0','2.0','125.0','7.0','234.0','23.0','1435.0',
        '25.0','1453.0','35.0','135.0','13245.0','235.0','1235.0','10.0','156.0'],
                            [1,4,10,15,10,8,12,4,5,
                                13,3,7,7,9,5,6,3,6,14,
                                12,6,11,2,8,7,9,5,13,
                                7,13,8,9,15,10,11,1,12], inplace = True)

        X['C7'].replace(['1.0','123.0','5.0','13.0','3.0','12.0','2.0','23.0','4.0','14.0','134.0',
        '124.0','1234.0','24.0','34.0'],
                                [1,6,5,4,3,3,2,5,4,5,8,
                                7,10,6,7], inplace = True)

        X['C13'].replace(['2.0','5.0','12345.0','2345.0','','1.0','245.0','24.0','15.0','25.0',
        '4.0','45.0','1245.0','125.0','12.0','2456.0','6.0','234.0','1235.0',
        '345.0','145.0','34.0','1234.0','235.0','3456.0','23.0','35.0','3.0',
        '246.0','124.0','456.0','1345.0','123456.0','123.0','12456.0','23456.0',
        '135.0','56.0','156.0','14.0','13.0','356.0'],
                                [2,5,15,14,np.nan,1,11,6,6,7,
                                4,9,12,8,3,17,6,9,11,
                                12,10,7,10,10,18,5,8,3,
                                12,7,15,13,21,6,18,20,
                                9,11,12,5,4,14], inplace = True)

        X['D2'].replace(['','2','23','1','4','3','1234','123','12','124','234','24','13','5',
        '12345','14','45','25','34','245','125','15','232','345'],
                                [np.nan,2,5,1,4,3,10,6,3,7,9,6,4,5,
                                15,5,9,7,7,11,8,6,7,12], inplace = True)

        X['D16'].replace(['12345.0','1234.0','0.0','1.0','134.0','234.0','3.0','1245.0','1345.0',
        '4.0','12.0','34.0','124.0','2345.0','2.0','345.0','23.0','5.0','123.0',
        '14.0','15.0','13.0','45.0','235.0','145.0','25.0','245.0','125.0','24.0',
        '135.0','243.0','1235.0','6.0','12354.0','1453.0','23456.0','1243.0',
        '43.0','32.0','112345.0','123456.0'],
                                [15,10,0,1,8,9,3,12,13,
                                4,3,7,7,14,2,12,5,5,6,
                                5,6,4,9,10,10,7,11,7,6,
                                9,9,11,6,15,13,20,10,
                                7,5,16,21], inplace = True)

        X['D17'].replace(['13.0','1234.0','1.0','0.0','3.0','123.0','2.0','134.0','4.0','234.0',
        '14.0','12.0','23.0','34.0','12345.0','43.0','5.0','143.0','1345.0',
        '124.0','45.0','145.0','135.0'],
                                [4,10,1,0,3,6,2,8,4,9,
                                5,3,5,7,15,7,5,8,13,
                                7,9,10,9], inplace = True)

        X['E4'].replace(['1.0','4.0','6.0','34.0','3.0','14.0','134.0','234.0','1234.0','2.0',
        '13.0','23.0','124.0','12.0','2345.0','24.0','5.0','123.0','135.0','45.0',
        '35.0','345.0','12345.0'],
                                [1,4,6,7,3,5,8,9,10,2,
                                4,5,7,3,14,6,5,6,9,9,
                                8,12,15], inplace = True)

        X['F5'].replace(['13.0','123.0','2.0','1.0','23.0','3.0','5.0','12.0','4.0','1234.0',
        '223.0','231.0','34.0','124.0','14.0','15.0','25.0','125.0'],
                                [4,6,2,1,5,3,5,3,4,10,
                                7,6,7,7,5,6,7,8], inplace = True)

        X['F10'].replace(['12345.0','345.0','14.0','4.0','1.0','3.0','124.0','13.0','134.0',
        '1234.0','12.0','0.0','5.0','2345.0','1345.0','34.0','23.0','1235.0',
        '45.0','15.0','123.0','24.0','135.0','6.0','35.0','2.0','145.0','234.0',
        '125.0','1245.0','25.0','1456.0','235.0','123456.0','245.0','13345.0',
        '23456.0'],
                                [15,12,5,4,1,3,7,4,8,
                                10,3,0,5,14,13,7,5,11,
                                9,6,6,6,9,6,8,2,10,10,
                                8,12,7,16,10,21,11,16,
                                20], inplace = True)

        X['F12'].replace(['1.0','5.0','','12345.0','245.0','3.0','1245.0','24.0','2345.0','25.0',
        '2.0','15.0','4.0','135.0','345.0','12.0','45.0','125.0','235.0','123.0',
        '6.0','124.0','35.0','13.0','145.0','1345.0','34.0','1235.0','14.0',
        '1234.0','234.0','23.0','123456.0','456.0','134.0','2245.0','246.0',
        '156.0'],
                                [1,5,np.nan,15,11,3,12,6,14,7,
                                2,6,4,9,12,3,9,8,10,6,
                                6,7,8,4,10,13,7,11,5,
                                10,9,5,21,15,8,13,12,
                                12], inplace = True)

        # X['vungtonthuong'].replace(['','"RM, RL, LM, LL"','"RU, RM, RL, LU, LM, LL"','"RU, RM, RL, LM, LL"',
        # '"RU, RM, RL"','"RU, RM, RL, lU, LM, LL"','"RL,  LL"','"RM, RL, LU"',
        # '"RM, RL"','"RM, RL,LL"','"RM, RL, LU, LM, LL"','"RU, RM, LM, LL"',
        # '"RM, RL, LL"'],
        #                         [np.nan,4,6,5,
        #                         3,6,2,3,2,3,5,4,
        #                         3], inplace = True)

        X['A6'].replace(['Da ket hon va dang chung song','Chua ket hon',
                                    'Li di','Goa','Li than'],
                                [2,0,1,0,1], inplace = True)

        X.replace(['', ' ', '.','khong biet'], np.nan, inplace = True)
        X['tsnoi'].replace(['khong', r'^\s*(?!0$).+\s*$'], [0, 1], regex=True, inplace = True)
        X['tsngoai'].replace(['khong', r'^\s*(?!0$).+\s*$'],  [0, 1], regex=True, inplace = True)
        X['rungthan'].replace(['giam', 'tang'], [-1, 1], regex=True, inplace = True)
        X['tsho'].replace(['khac','lien tuc','tung con'], [0,2,1], inplace = True)
        X['vitridau'].replace(['hai ben nguc','nguc phai','dau sau xuong uc','nguc trai'], [3,1,4,2], inplace = True)
        X['longnguc'].replace(['can doi','khong can doi'], [1,0], inplace = True)
        X['khoangls'].replace(['Binh thuong', 'Gian rong', 'Hep'], [0,1,-1], inplace = True)
        X['go'].replace(['go duc'], [1], inplace = True)
        X['riraopn'].replace(['giam','khac'], [-1,1], inplace = True)
        X['B7'].replace(['hut hang ngay','hut 1 tuan 1 lan','hiem khi hut','hut 1 thang 1 lan'],[2,1,-1,0], inplace = True) 

        X.replace(['binh thuong','tot','rat tot','khong tot'],[0,1,2,-1], inplace = True)
        X.replace(['khong thoai mai','thoai mai','rat khong thoai mai','rat thoai mai'],[-1,1,-2,2], inplace = True)
        X.replace(['nhieu','trung binh','rat nhieu','it','rat it'],[1,0,2,-1,-2], inplace = True)
        X.replace(['nguy hiem','rat nguy hiem','khong nguy hiem'],[1,2,0], inplace = True)
        X.replace(['de mac','rat de mac','khong the mac'],[0,1,-1], inplace = True)
        X.replace(['rat thuong xuyen','thuong xuyen','thinh thoang','hiem khi','khong su dung'],[2,1,0,-1,-2], inplace = True)
        X.replace(['khong','co','ko'],[0,1,0], inplace = True)
        X.replace(['ban ngay','ca ngay va dem','ban dem'],[0,2,1], inplace = True)

        encoded_labels = []
        for col in X.columns:
            try:
                X[col] = X[col].astype(float)
            except Exception:
                encoded_labels.append(col)
                try:
                    X[col] = X[col].astype(str)
                except Exception:
                    pass
        
        X['sdt'] = X['sdt'].astype(str)
        encoded_labels.append('sdt')

        #Encode data by OrdinalEncoder
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        oe.fit(self.not_encoded_df[encoded_labels])
        self.df[encoded_labels] = oe.transform(self.df[encoded_labels])

    def normalize(self, X):
        for column in X.columns:
            if (self.max_values[column] - self.MID_LIST[column]) != 0:
                X[column] = (X[column] - self.MID_LIST[column])  / (self.max_values[column] - self.MID_LIST[column])

    def create_graph_list(self, X):
        graph = []
        columns = X.columns.tolist()
        relation_ship = [['cviec1','tgian1'],['cviec2','tgian2'],['tuoinghe','A9a','A9b','nampx'],
                        ['hutthuoc','slthuoc'],['tiensuhh','benhhh'],['ho','tdho','tsho'],
                        ['khacdom','loaidom','tdkhacdo'],['khotho','mdkhotho','tdkhotho'],
                        ['daunguc','vitridau','daulan','tcdau','tgdau','ytodau'],
                        ['sutcan','socansut','tgsut'],['fvc','fev1']]
        
        for tmp_i in range(len(relation_ship)):
            for tmp_j in range(len(relation_ship[tmp_i])):
                relation_ship[tmp_i][tmp_j] = columns.index(relation_ship[tmp_i][tmp_j])
        
        for i in range(X.shape[0]):
            source = []
            target = []
            value = []
            indexing = []
            
            for j in range (0, X.shape[1]):
                if not np.isnan(X.iloc[i][j]):
                    source.append(0)
                    target.append(j)
                    value.append(X.iloc[i][j])
                    indexing.append(j)
                    
            for j in range(len(relation_ship)):
                flg = True
                for xx in relation_ship[j]:
                    if np.isnan(X.iloc[i][xx]):
                        flg = False
                        break
                if flg:
                    for xx in relation_ship[j]:
                        source.append(relation_ship[j][0])
                        target.append(xx) 
                    
            sg_tmp = pd.DataFrame(
                {"source": source, "target": target}
            )
            sg_tmp_data = pd.DataFrame(
                {"value": value}, 
                index = indexing
            )
            
            sg = StellarGraph(sg_tmp_data, sg_tmp)
            graph.append(sg)
            
        return graph
    
    def get_feature_list(self):
        if self.COLUMNS_NAME == None:
            open_file = open(COLUMNS_NAME_FILE_NAME, "rb")
            self.COLUMNS_NAME = pickle.load(open_file)
            open_file.close()
        feature_list = []
        columns = self.df.columns
        for i in range(0, len(columns)):
            fet = {}
            fet['name'] = self.COLUMNS_NAME[i]
            fet['col'] = columns[i]
            fet['id'] = i
            feature_list.append(fet)
        
        return feature_list