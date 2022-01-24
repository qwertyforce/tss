import uvicorn
if __name__ == '__main__':
    uvicorn.run('nn_web:app', host='127.0.0.1', port=33334, log_level="info")

from os import listdir
from typing import Optional
import torch
from pydantic import BaseModel
from fastapi import FastAPI, File,Form, HTTPException, Response, status
import numpy as np
from PIL import Image
import torch
import timm
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

IMAGE_PATH="./../../../public/images"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model('beit_base_patch16_224_in22k', pretrained=True)
model.head=torch.nn.Identity()
model.eval()
model.to(device)

import sqlite3
import io
conn = sqlite3.connect('NN_features.db')
conn_orig = sqlite3.connect('NN_features_orig.db')


dim=768
import faiss
sub_index = faiss.IndexFlatL2(dim)
index_id_map = faiss.IndexIDMap2(sub_index)
# import hnswlib
# index = hnswlib.Index(space='l2', dim=dim)
# index.init_index(max_elements=1_000_000, ef_construction=200, M=32)

import pickle
pca_w_file = Path("./pca_w.pkl")
pca = None
if pca_w_file.is_file():
    with open(pca_w_file, 'rb') as pickle_file:
        pca = pickle.load(pickle_file)

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


def get_all_data_iterator(arraysize=10000):
    cursor = conn.cursor()
    query = '''
        SELECT id, nn_features
        FROM nn_table
        '''
    cursor.execute(query)
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        yield results


def init_index():
    for result in tqdm(get_all_data_iterator(10000)):
        ids = [x[0] for x in result]
        features = [convert_array(x[1]) for x in result]
        # features = Parallel(n_jobs=1)(delayed(convert_array)(feature) for feature in features)
        ids=np.int64(ids)
        features=np.array(features,dtype=np.float32)
        index_id_map.add_with_ids(features,ids)
    print("Index is ready")

def read_img_buffer(image_data):
    img = Image.open(io.BytesIO(image_data))
    img=img.convert('L').convert('RGB') #GREYSCALE
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')
    return img

def get_features(image_buffer):
    image=read_img_buffer(image_buffer)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature_vector = model(image).cpu().numpy()[0]
    feature_vector/=np.linalg.norm(feature_vector)
    return feature_vector

def create_table():
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

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def add_descriptor(id,nn_features):
    cursor = conn.cursor()
    query = '''INSERT INTO nn_table(id, nn_features) VALUES (?,?)'''
    cursor.execute(query,(id,nn_features))
    conn.commit()

def add_descriptor_orig(id,nn_features):
    cursor = conn_orig.cursor()
    query = '''INSERT INTO nn_table(id, nn_features) VALUES (?,?)'''
    cursor.execute(query,(id,nn_features))
    conn_orig.commit()


def sync_db():
    ids_in_db=set(get_all_ids())
    file_names=listdir(IMAGE_PATH)
    for file_name in file_names:
        file_id=int(file_name[:file_name.index('.')])
        if file_id in ids_in_db:
            ids_in_db.remove(file_id)
    for id in ids_in_db:
        delete_descriptor_by_id(id)   #Fix this
        print(f"deleting {id}")
    print("db synced")


def get_aqe_vector(feature_vector):
    top_n_query=5
    _, I = index_id_map.search(feature_vector, top_n_query)
    alpha=1
    top_features=[]
    for i in range(top_n_query):
        top_features.append(index_id_map.reconstruct(int(list(I[0])[i])).flatten())
    new_feature=[]
    for i in range(dim):
        _sum=0
        for j in range(top_n_query):
            _sum+=top_features[j][i] * np.dot(feature_vector, top_features[j].T)**alpha
        new_feature.append(_sum)

    new_feature=np.array(new_feature)
    new_feature/=np.linalg.norm(new_feature)
    new_feature=new_feature.astype(np.float32).reshape(1,-1)
    return new_feature

def nn_find_similar(feature_vector, k, distance_threshold):
    new_feature_vector=get_aqe_vector(feature_vector)
    print(new_feature_vector)
    if k is not None:
        D, I = index_id_map.search(new_feature_vector, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        _, D, I = index_id_map.range_search(new_feature_vector, distance_threshold)

    res=[{"image_id":int(I[i]),"distance":float(D[i])} for i in range(len(D))]
    res = sorted(res, key=lambda x: x["distance"]) 
    return res

app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.post("/calculate_nn_features")
async def calculate_NN_features_handler(image: bytes = File(...),image_id: str = Form(...)):
    try:
        image_id=int(image_id)
        features=get_features(image)
        add_descriptor_orig(image_id,adapt_array(features.astype(np.float32)))
        if pca:
            features=pca.transform(features.reshape(1,-1))[0]
            features/=np.linalg.norm(features)
        features=features.astype(np.float32)
        add_descriptor(image_id,adapt_array(features))
        # index.add_items(features,[image_id])
        index_id_map.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
        return Response(status_code=status.HTTP_200_OK)
    except:
        raise HTTPException(status_code=500, detail="Can't calculate nn features")

class Item_image_id(BaseModel):
    image_id: int

@app.post("/delete_nn_features")
async def delete_nn_features_handler(item:Item_image_id):
    delete_descriptor_by_id(item.image_id)
    res = index_id_map.remove_ids(np.int64([item.image_id]))
    if res == 0: #nothing to delete
        print(f"err: no image with id {item.image_id}")
    return Response(status_code=status.HTTP_200_OK)
    # try:
    #     # index.mark_deleted(item.image_id)
    #     index_id_map.remove_ids(np.int64([item.image_id]))
    # except RuntimeError:
    #     print(f"err: no image with id {item.image_id}")
    # return Response(status_code=status.HTTP_200_OK)


class Item_image_id_k(BaseModel):
    image_id: int
    k: Optional[str] = None
    distance_threshold: Optional[str] = None

@app.post("/nn_get_similar_images_by_id")
async def get_similar_images_by_id_handler(item: Item_image_id_k):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        if item.k:
            k = int(k)
        if item.distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = index_id_map.reconstruct(item.image_id).reshape(1,-1)
        print(target_features)
        #target_features = index.get_items([item.image_id])
        results = nn_find_similar(target_features,k, distance_threshold)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Can't get get_similar_images_by_id_handler")

@app.post("/nn_get_similar_images_by_image_buffer")
async def nn_get_similar_images_by_image_buffer_handler(image: bytes = File(...), k: Optional[str] = Form(None), distance_threshold: Optional[str] = Form(None)):
    try:
        if k:
            k = int(k)
        if distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features=get_features(image)
        if pca:
            target_features=pca.transform(target_features.reshape(1,-1))[0]
            target_features/=np.linalg.norm(target_features)
        target_features=target_features.astype(np.float32).reshape(1,-1)
        results = nn_find_similar(target_features, k, distance_threshold)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Can't get nn_get_similar_images_by_image_buffer_handler")

print(__name__)
if __name__ == 'nn_web':
    create_table()
    # sync_db()
    init_index()
