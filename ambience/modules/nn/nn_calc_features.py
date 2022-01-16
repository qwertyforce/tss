import pickle
import torch
import timm
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('beit_base_patch16_224_in22k', pretrained=True)
model.head=torch.nn.Identity()
model.eval()
model.to(device)

from os import listdir
from pathlib import Path
import numpy as np
from PIL import Image
import sqlite3
import io
from tqdm import tqdm

conn = sqlite3.connect('NN_features.db')
conn_orig_db = sqlite3.connect('NN_features_orig.db')
# IMAGE_PATH="./../../../public/images
IMAGE_PATH="./../../../test_images"

def create_table(conn):
	cursor = conn.cursor()
	query = '''
	    CREATE TABLE IF NOT EXISTS nn_table(
	    	id INTEGER NOT NULL UNIQUE PRIMARY KEY, 
	    	nn_features BLOB NOT NULL
	    )
	'''
	cursor.execute(query)
	conn.commit()

def check_if_exists_by_id(id):
    cursor = conn.cursor()
    query = '''SELECT EXISTS(SELECT 1 FROM nn_table WHERE id=(?))'''
    cursor.execute(query,(id,))
    all_rows = cursor.fetchone()
    return all_rows[0] == 1    

def delete_descriptor_by_id(id):
	cursor = conn.cursor()
	query = '''DELETE FROM nn_table WHERE id=(?)'''
	cursor.execute(query,(id,))
	conn.commit()

def get_all_ids():
    cursor = conn.cursor()
    query = '''SELECT id FROM nn_table'''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:el[0],all_rows))

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def sync_db():
    file_names=listdir(IMAGE_PATH)
    ids_in_db=set(get_all_ids())

    for file_name in file_names:
        file_id=int(file_name[:file_name.index('.')])
        if file_id in ids_in_db:
            ids_in_db.remove(file_id)
    for id in ids_in_db:
        delete_descriptor_by_id(id)   #Fix this
        print(f"deleting {id}")

_transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def transform(im):
  desired_size = 224
  old_size = im.size  # old_size[0] is in (width, height) format
  ratio = float(desired_size)/max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])
  im = im.resize(new_size, Image.ANTIALIAS)
  new_im = Image.new("RGB", (desired_size, desired_size))
  new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
  return _transform(new_im)

  
def get_features(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model(image).cpu().numpy()[0]
    feature_vector/=np.linalg.norm(feature_vector)
    return feature_vector

file_names=listdir(IMAGE_PATH)
create_table(conn)
create_table(conn_orig_db)
# sync_db()

new_images=[]
for file_name in file_names:
    file_id=int(file_name[:file_name.index('.')])
    if check_if_exists_by_id(file_id):
        continue
    new_images.append(file_name)

def calc_nn_features(file_name):
    file_id=int(file_name[:file_name.index('.')])
    img_path=IMAGE_PATH+"/"+file_name
    try:
        query_image=Image.open(img_path)
    except:
        print(f'error reading {img_path}')
        return None
    image_features=get_features(query_image) 
    # image_features_bin=adapt_array(image_features)
    return (file_id,image_features)


pca_w_file = Path("./pca_w.pkl")
pca = None
if pca_w_file.is_file():
    with open(pca_w_file, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

new_images=[new_images[i:i + 5000] for i in range(0, len(new_images), 5000)]
for batch in new_images:
    batch_features=[]
    for file_name in tqdm(batch):
        batch_features.append(calc_nn_features(file_name))
    batch_features= [i for i in batch_features if i] #remove None's
    print("pushing data to db")
    
    batch_features_orig=[( x[0], adapt_array(x[1].astype(np.float32)) ) for x in batch_features]
    conn_orig_db.executemany('''INSERT INTO nn_table(id, nn_features) VALUES (?,?)''', batch_features_orig) 

    if pca is not None:
        image_features=[x[1] for x in batch_features]
        image_features=pca.transform(image_features) #pca whitening
        image_features/=np.linalg.norm(image_features) # l2 norm
        pca_whitened_batch_features=[]
        for i in range(len(batch_features)):
            pca_whitened_batch_features.append( (batch_features[i][0],adapt_array(image_features[i].astype(np.float32)) ) )
        conn.executemany('''INSERT INTO nn_table(id, nn_features) VALUES (?,?)''', pca_whitened_batch_features)
    conn_orig_db.commit()
    conn.commit()

import subprocess
def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def get_all_data():
    cursor = conn_orig_db.cursor()
    query = '''
    SELECT id, nn_features
    FROM nn_table
    '''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    # return list(map(lambda el:{"image_id":el[0],"features":convert_array(el[1])},all_rows))
    return ( list(map(lambda el:el[0],all_rows)),list(map(lambda el:convert_array(el[1]),all_rows)) )



if pca is None:
    print("looks like initial calc, running pca_w")
    subprocess.call(['python', 'calc_pca_w.py'])
    with open(pca_w_file, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)
    image_ids,image_features=get_all_data()
    image_features=pca.transform(image_features) #pca whitening
    image_features/=np.linalg.norm(image_features) # l2 norm

    pca_whitened_batch_features=[]
    for i in range(len(image_ids)):
         pca_whitened_batch_features.append((image_ids[i],adapt_array(image_features[i].astype(np.float32))))
    conn.executemany('''INSERT INTO nn_table(id, nn_features) VALUES (?,?)''', pca_whitened_batch_features)  
    conn.commit()  
    


    