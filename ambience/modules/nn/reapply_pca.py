from pathlib import Path
import numpy as np
import sqlite3
import io
import pickle
conn = sqlite3.connect('NN_features.db')
conn_new_db = sqlite3.connect('NN_features_new.db')
conn_orig = sqlite3.connect('NN_features_orig.db')



def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def get_all_ids():
    cursor = conn.cursor()
    query = '''SELECT id FROM nn_table'''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:el[0],all_rows))

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def create_table(connect):
	cursor = connect.cursor()
	query = '''
	    CREATE TABLE IF NOT EXISTS nn_table(
	    	id INTEGER NOT NULL UNIQUE PRIMARY KEY, 
	    	nn_features BLOB NOT NULL
	    )
	'''
	cursor.execute(query)
	connect.commit()

def get_orig_nn_features_by_id(id):
    cursor = conn_orig.cursor()
    query = '''
    SELECT nn_features
    FROM nn_table
    WHERE id = (?)
    '''
    cursor.execute(query, (id,))
    all_rows = cursor.fetchone()
    return convert_array(all_rows[0])

import pickle
pca_w_file = Path("./pca_w.pkl")
pca = None
if pca_w_file.is_file():
    with open(pca_w_file, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

if pca is None:
    print("no pca_w.pkl found. exiting...")
    exit()

ids=get_all_ids()
features=[get_orig_nn_features_by_id(id) for id in ids]
features=pca.transform(features)
features/=np.linalg.norm(features)
pca_whitened_batch_features=[]
for i in range(len(ids)):
    pca_whitened_batch_features.append((ids[i],adapt_array(features[i].astype(np.float32))))
create_table(conn_new_db)
conn_new_db.executemany('''INSERT INTO nn_table(id, nn_features) VALUES (?,?)''', pca_whitened_batch_features)  
conn_new_db.commit()  
