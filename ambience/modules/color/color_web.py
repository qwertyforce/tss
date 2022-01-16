import uvicorn
if __name__ == '__main__':
    uvicorn.run('color_web:app', host='127.0.0.1', port=33335, log_level="info")

from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, Response, status
import faiss
from os import listdir
import numpy as np
from tqdm import tqdm
import cv2
import sqlite3
import io
conn = sqlite3.connect('rgb_histograms_120k.db')
IMAGE_PATH = "./../../../public/images"
sub_index = faiss.IndexFlat(512, faiss.METRIC_L1)
index_id_map = faiss.IndexIDMap2(sub_index)


def init_index():
    # all_ids = get_all_ids()
    # for image_id in tqdm(all_ids):
    #     features = convert_array(get_rgb_histogram_by_id(image_id))
    #     index_id_map.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
    image_data=get_all_data()
    features=[]
    ids=[]
    for image in image_data:
        features.append(image['features'])
        ids.append(image['image_id'])

    ids=np.int64(ids)
    features=np.array(features,dtype=np.float32)
    if(len(features)!=0):
        index_id_map.add_with_ids(features,ids)
    print("Index is ready")


def read_img_file(image_data):
    return np.fromstring(image_data, np.uint8)


def get_rgb_histogram_by_id(id):
    cursor = conn.cursor()
    query = '''
    SELECT rgb_histogram
    FROM rgb_hists
    WHERE id = (?)
    '''
    cursor.execute(query, (id,))
    all_rows = cursor.fetchone()
    return all_rows[0]


def get_features(image_buffer):
    query_image = cv2.cvtColor(cv2.imdecode(read_img_file(image_buffer), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    query_hist_combined = cv2.calcHist([query_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    query_hist_combined = query_hist_combined.flatten()
    query_hist_combined = query_hist_combined*10000000
    query_hist_combined = np.divide(query_hist_combined, query_image.shape[0]*query_image.shape[1], dtype=np.float32)
    return query_hist_combined


def create_table():
    cursor = conn.cursor()
    query = '''
	    CREATE TABLE IF NOT EXISTS rgb_hists(
	    	id INTEGER NOT NULL UNIQUE PRIMARY KEY, 
	    	rgb_histogram BLOB NOT NULL
	    )
	'''
    cursor.execute(query)
    conn.commit()


def check_if_exists_by_id(id):
    cursor = conn.cursor()
    query = '''SELECT EXISTS(SELECT 1 FROM rgb_hists WHERE id=(?))'''
    cursor.execute(query, (id,))
    all_rows = cursor.fetchone()
    return all_rows[0] == 1


def delete_descriptor_by_id(id):
    cursor = conn.cursor()
    query = '''DELETE FROM rgb_hists WHERE id=(?)'''
    cursor.execute(query, (id,))
    conn.commit()


def get_all_ids():
    cursor = conn.cursor()
    query = '''SELECT id FROM rgb_hists'''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el: el[0], all_rows))

def get_all_data():
    cursor = conn.cursor()
    query = '''
    SELECT id, rgb_histogram
    FROM rgb_hists
    '''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:{"image_id":el[0],"features":convert_array(el[1])},all_rows))


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def add_descriptor(id, rgb_histogram):
    cursor = conn.cursor()
    query = '''INSERT INTO rgb_hists(id, rgb_histogram) VALUES (?,?)'''
    cursor.execute(query, (id, rgb_histogram))
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


def hist_similarity_search(target_features, k, distance_threshold):
    if k is not None:
        D, I = index_id_map.search(target_features, k)
        D = D.flatten()
        I = I.flatten()
    elif distance_threshold is not None:
        start_k=min(index_id_map.ntotal, 100)
        while True:
            D, I = index_id_map.search(target_features,start_k)
            D = D.flatten()
            I = I.flatten()
            if max(D) < distance_threshold:
                if(start_k == index_id_map.ntotal):
                    break
                start_k*=2
            else:
                indexes=np.where(D < distance_threshold)[0]
                D=D[indexes]
                I=I[indexes]
                break
            if(start_k > index_id_map.ntotal):
                    break
            print(start_k)

    res=[{"image_id":int(I[i]),"distance":float(D[i])} for i in range(len(D))]

    return res


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/calculate_color_features")
async def calculate_color_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    features = get_features(image)
    add_descriptor(int(image_id), adapt_array(features))
    index_id_map.add_with_ids(features.reshape(1,-1), np.int64([image_id]))
    return Response(status_code=status.HTTP_200_OK)


class Item_image_id(BaseModel):
    image_id: int
    k: Optional[str] = None
    distance_threshold: Optional[str] = None

@app.post("/color_get_similar_images_by_id")
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

        target_features = index_id_map.reconstruct(item.image_id).reshape(1,-1)
        similar = hist_similarity_search(target_features, k, distance_threshold)
        return similar
    except:
        raise HTTPException(
            status_code=500, detail="Image with this id is not found")


@app.post("/color_get_similar_images_by_image_buffer")
async def color_get_similar_images_by_image_buffer_handler(image: bytes = File(...), k: Optional[str] = Form(None), distance_threshold: Optional[str] = Form(None)):
    try:
        if k:
            k = int(k)
        if distance_threshold:
            distance_threshold = float(distance_threshold)
        if (k is None) == (distance_threshold is None):
            raise HTTPException(status_code=500, detail="both k and distance_threshold present")

        target_features=get_features(image).reshape(1,-1)
        results = hist_similarity_search(target_features, k, distance_threshold)
        return results
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Can't get color_get_similar_images_by_image_buffer")


@app.post("/delete_color_features")
async def delete_color_features_handler(item: Item_image_id):
    delete_descriptor_by_id(item.image_id)
    index_id_map.remove_ids(np.int64([item.image_id]))
    return Response(status_code=status.HTTP_200_OK)

print(__name__)

if __name__ == 'color_web':
    create_table()
    # sync_db()
    init_index()
