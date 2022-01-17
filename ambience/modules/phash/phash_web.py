import uvicorn
if __name__ == '__main__':
    uvicorn.run('phash_web:app', host='127.0.0.1', port=33336, log_level="info")

import faiss
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, Response, status, HTTPException
from os import listdir
import numpy as np
from scipy.fft import dct
from numba import jit
import cv2
import sqlite3
import io
conn = sqlite3.connect('phashes.db')
index = None
IMAGE_PATH = "./../../../public/images"
from tqdm import tqdm

all_data=None
def init_index():
    global index,all_data
    try:
        index = faiss.read_index_binary("trained.index")
    except:
        d = 72*8
        quantizer = faiss.IndexBinaryFlat(d)
        index = faiss.IndexBinaryIVF(quantizer, d, 1)
        index.nprobe = 1
        index.train(np.array([np.zeros(72)], dtype=np.uint8))
    all_data = get_all_data()
    image_ids = np.array([np.int64(x[0]) for x in all_data])
    phashes = np.array([x[1] for x in all_data])
    if len(all_data) != 0:
        print(phashes.shape)
        index.add_with_ids(phashes, image_ids)
    print("Index is ready")


def read_img_file(image_data):
    return np.fromstring(image_data, np.uint8)


@jit(cache=True, nopython=True)
def bit_list_to_72_uint8(bit_list_576):
    uint8_arr = []
    for i in range(len(bit_list_576)//8):
        bit_list = []
        for j in range(8):
            if(bit_list_576[i*8+j] == True):
                bit_list.append(1)
            else:
                bit_list.append(0)
        uint8_arr.append(bit_list_to_int(bit_list))
    return np.array(uint8_arr, dtype=np.uint8)


@jit(cache=True, nopython=True)
def bit_list_to_int(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


@jit(cache=True, nopython=True)
def diff(dct, hash_size):
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()


def fast_phash(resized_image, hash_size):
    dct_data = dct(dct(resized_image, axis=0), axis=1)
    return diff(dct_data, hash_size)


def get_phash(image_buffer, hash_size=24, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    query_image = cv2.imdecode(read_img_file(image_buffer), cv2.IMREAD_GRAYSCALE)
    query_image = cv2.resize(query_image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    bit_list_576 = fast_phash(query_image, hash_size)
    phash = bit_list_to_72_uint8(bit_list_576)
    return phash


def get_phash_and_mirrored_phash(image_buffer, hash_size=24, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    query_image = cv2.imdecode(read_img_file(image_buffer), cv2.IMREAD_GRAYSCALE)
    query_image = cv2.resize(query_image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    mirrored_query_image = cv2.flip(query_image, 1)
    bit_list_576 = fast_phash(query_image, hash_size)
    bit_list_576_mirrored = fast_phash(mirrored_query_image, hash_size)
    phash = bit_list_to_72_uint8(bit_list_576)
    mirrored_phash = bit_list_to_72_uint8(bit_list_576_mirrored)
    return np.array([phash, mirrored_phash])


def create_table():
    cursor = conn.cursor()
    query = '''
	    CREATE TABLE IF NOT EXISTS phashes(
	    	id INTEGER NOT NULL UNIQUE PRIMARY KEY, 
	    	phash BLOB NOT NULL
	    )
	'''
    cursor.execute(query)
    conn.commit()


def check_if_exists_by_id(id):
    cursor = conn.cursor()
    query = '''SELECT EXISTS(SELECT 1 FROM phashes WHERE id=(?))'''
    cursor.execute(query, (id,))
    all_rows = cursor.fetchone()
    return all_rows[0] == 1


def delete_descriptor_by_id(id):
    cursor = conn.cursor()
    query = '''DELETE FROM phashes WHERE id=(?)'''
    cursor.execute(query, (id,))
    conn.commit()


def get_all_ids():
    cursor = conn.cursor()
    query = '''SELECT id FROM phashes'''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el: el[0], all_rows))


def get_all_data():
    cursor = conn.cursor()
    query = '''
    SELECT id, phash
    FROM phashes
    '''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el: (el[0], convert_array(el[1])), all_rows))


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def add_descriptor(id, phash):
    cursor = conn.cursor()
    query = '''INSERT INTO phashes(id, phash) VALUES (?,?)'''
    cursor.execute(query, (id, phash))
    conn.commit()


def sync_db():
    ids_in_db = set(get_all_ids())
    file_names = listdir(IMAGE_PATH)
    for file_name in file_names:
        file_id = int(file_name[:file_name.index('.')])
        if file_id in ids_in_db:
            ids_in_db.remove(file_id)
    for id in ids_in_db:
        delete_descriptor_by_id(id)  # Fix this
        print(f"deleting {id}")
    print("db synced")


def phash_reverse_search(target_features,k,distance_threshold):
    if k is not None:
        D, I = index.search(target_features, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        _, D, I = index.range_search(target_features, distance_threshold)
    
    _, indexes = np.unique(I, return_index=True)
    res=[{"image_id":int(I[idx]), "distance":int(D[idx])} for idx in indexes]
    res = sorted(res, key=lambda x: x["distance"])    
    return res


app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"}


class Item_image_id(BaseModel):
    image_id: int
    k: Optional[str] = None
    distance_threshold: Optional[str] = None

@app.post("/phash_get_similar_images_by_id")
async def color_get_similar_images_by_id_handler(item: Item_image_id):
    try:
        k=item.k
        distance_threshold=item.distance_threshold
        if item.k:
            k = int(k)
        if item.distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features = index.reconstruct(item.image_id).reshape(1,-1)
        similar = phash_reverse_search(target_features,k,distance_threshold)
        return similar
    except:
        raise HTTPException(
            status_code=500, detail="Image with this id is not found")

@app.post("/phash_get_similar_images_by_image_buffer")
async def phash_reverse_search_handler(image: bytes = File(...), k: Optional[str] = Form(None), distance_threshold: Optional[str] = Form(None)):
    if k:
        k=int(k)
    if distance_threshold:
       distance_threshold=int(distance_threshold)
    if (k is None) == (distance_threshold is None): 
        raise HTTPException(status_code=500, detail="both k and distance_threshold present")
    target_features = get_phash_and_mirrored_phash(image) #TTA
    similar = phash_reverse_search(target_features,k,distance_threshold)
    print(similar)
    return similar


@app.post("/calculate_phash_features")
async def calculate_phash_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    features = get_phash(image)
    add_descriptor(int(image_id), adapt_array(features))
    index.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
    return Response(status_code=status.HTTP_200_OK)

@app.post("/delete_phash_features")
async def delete_hist_features_handler(item: Item_image_id):
    delete_descriptor_by_id(item.image_id)
    index.remove_ids(np.int64([item.image_id]))
    return Response(status_code=status.HTTP_200_OK)

print(__name__)
if __name__ == 'phash_web':
    create_table()
    # sync_db()
    init_index()
