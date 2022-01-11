import numpy as np
import sqlite3
import io
import pickle
conn = sqlite3.connect('NN_features_orig.db')
from sklearn.decomposition import PCA
pca = PCA(n_components=768, whiten=True, copy=False)

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def get_all_data():
    cursor = conn.cursor()
    query = '''
    SELECT nn_features
    FROM nn_table
    '''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:convert_array(el[0]),all_rows))


features=get_all_data()
pca.fit(features)
with open('pca_w.pkl', 'wb') as handle:
    pickle.dump(pca, handle)