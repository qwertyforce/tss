import uvicorn
if __name__ == '__main__':
    uvicorn.run('text_web:app', host='127.0.0.1', port=33339, log_level="info")

from pydantic import BaseModel
from fastapi import FastAPI, File, Form, HTTPException, Response, status
# import faiss
from os import listdir
import numpy as np
from tqdm import tqdm
import cv2
import sqlite3
import json
import math
from paddleocr import PaddleOCR
ocr_ru = PaddleOCR(use_angle_cls=True, lang="ru",show_log = False)
ocr_en = PaddleOCR(use_angle_cls=True, lang="en",show_log = False)

conn = sqlite3.connect('ocr_text.db')

from rapidfuzz import process, fuzz
import re 
# IMAGE_PATH = "./../../../public/images"
# sub_index = faiss.IndexFlat(512, faiss.METRIC_L1)
# index_id_map = faiss.IndexIDMap2(sub_index)

IMG_ID_TXT_ARR={}

def get_all_data():
    cursor = conn.cursor()
    query = '''
    SELECT id, text_arr
    FROM ocr_text
    '''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el:{"image_id":el[0],"text_arr":el[1]},all_rows))

def init_index():
    image_data=get_all_data()
    print(f'Images in db: {len(image_data)}')
    for el in tqdm(image_data):
        text_arr=json.loads(el["text_arr"])
        IMG_ID_TXT_ARR[el["image_id"]]=text_arr
    print(IMG_ID_TXT_ARR[261764])
    print("Index is ready")

def get_text_by_image_id(id):
    cursor = conn.cursor()
    query = '''
    SELECT text_arr
    FROM ocr_text
    WHERE id = (?)
    '''
    cursor.execute(query, (id,))
    all_rows = cursor.fetchone()
    return all_rows[0]

def create_table():
    cursor = conn.cursor()
    query = '''
	    CREATE TABLE IF NOT EXISTS ocr_text(
	    	id INTEGER NOT NULL UNIQUE PRIMARY KEY, 
	    	text_arr TEXT NOT NULL
	    )
	'''
    cursor.execute(query)
    conn.commit()


def check_if_exists_by_id(id):
    cursor = conn.cursor()
    query = '''SELECT EXISTS(SELECT 1 FROM ocr_text WHERE id=(?))'''
    cursor.execute(query, (id,))
    all_rows = cursor.fetchone()
    return all_rows[0] == 1


def delete_descriptor_by_id(id):
    cursor = conn.cursor()
    query = '''DELETE FROM ocr_text WHERE id=(?)'''
    cursor.execute(query, (id,))
    conn.commit()

def get_all_ids():
    cursor = conn.cursor()
    query = '''SELECT id FROM ocr_text'''
    cursor.execute(query)
    all_rows = cursor.fetchall()
    return list(map(lambda el: el[0], all_rows))

def add_descriptor(id, text_arr):
    cursor = conn.cursor()
    query = '''INSERT INTO ocr_text(id, text_arr) VALUES (?,?)'''
    cursor.execute(query, (id, text_arr))
    conn.commit()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def convert_to_words_arr(text_arr):
    words_arr=[]
    for entry in text_arr:
        word=entry[1][0]
        # words_arr.append(word)
        word = word.strip()
        words_split= word.split()
        words_arr.extend(words_split)
    return words_arr
# def sync_db():
#     ids_in_db = set(get_all_ids())
#     file_names = listdir(IMAGE_PATH)
#     for file_name in file_names:
#         file_id = int(file_name[:file_name.index('.')])
#         if file_id in ids_in_db:
#             ids_in_db.remove(file_id)
#     for id in ids_in_db:
#         delete_descriptor_by_id(id)  # Fix this
#         print(f"deleting {id}")
#     print("db synced")

def find_similar_by_text(query_text_arr):
    pass

def read_img_file(image_data):
    return np.fromstring(image_data, np.uint8)

def resize_img_to_threshold(img):
    height,width=img.shape
    threshold=2000*1500
    if height*width>threshold:
        k=math.sqrt(height*width/(threshold))
        img=cv2.resize(img, (round(width/k),round(height/k)), interpolation=cv2.INTER_LINEAR)
    return img

def get_ocr_text(image_buffer):
    query_image = cv2.imdecode(read_img_file(image_buffer),cv2.IMREAD_GRAYSCALE)
    query_image = resize_img_to_threshold(query_image)

    result_ru = ocr_ru.ocr(query_image, cls=True)
    result_en = ocr_en.ocr(query_image, cls=True)
    final_words = []
    for txt in result_ru:
        flag1 = False
        for _txt in result_en:
            if str(txt[0]) == str(_txt[0]):
                flag1 = True
                if txt[1][1] > _txt[1][1]:
                    final_words.append(txt)
                else:
                    final_words.append(_txt)
                break
        if flag1 == False:
            final_words.append(txt)
    # print(final_words)
    for txt in result_en:
        flag1 = False
        for _txt in final_words:
            if str(txt[0]) == str(_txt[0]):
                flag1 = True
                break
        if flag1 == False:
            final_words.append(txt)
    print(final_words)
    return final_words


def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))


from fonetika.metaphone import RussianMetaphone
METAPHONE_RU=RussianMetaphone()

from abydos import phonetic,distance
METAPHONE_ENG=phonetic.Metaphone()

from transliterate import translit
import rapidfuzz

def text_find_similar(text_arr):
    words_arr = convert_to_words_arr(text_arr)
    # query_text_en="adidas"
    # query_text_ru=translit(query_text_en,"ru",reversed=False)
    # metaphone_ru=METAPHONE_RU.transform(query_text_ru)
    # metaphone_eng=METAPHONE_ENG.encode(query_text_en)
    similar=set()
    for key,value in tqdm(IMG_ID_TXT_ARR.items()): #key == image_id
        # print(image)
        for target_word in words_arr:
            target_word_is_cyrillic=has_cyrillic(target_word)
            db_words=convert_to_words_arr(value)
            for db_word in db_words:
                w_ratio_metaphone_en=None
                w_ratio_metaphone_ru=None
                w_ratio_orig=None
                if has_cyrillic(db_word) == target_word_is_cyrillic:
                    w_ratio_orig=rapidfuzz.string_metric.normalized_levenshtein(target_word,db_word,weights=(2,2,1))
                    if target_word_is_cyrillic==True:
                        target_word_metaphone_ru = METAPHONE_RU.transform(target_word)
                        db_word_metaphone_ru = METAPHONE_RU.transform(db_word)
                        w_ratio_metaphone_ru=fuzz.WRatio(target_word_metaphone_ru,db_word_metaphone_ru)
                    else:
                        target_word_arr=target_word.strip().split()
                        target_word_metaphone_eng=""
                        for w in target_word_arr:
                            target_word_metaphone_eng+=METAPHONE_ENG.encode(w)

                        db_word_arr=db_word.strip().split()
                        db_word_metaphone_eng=""
                        for w in db_word_arr:
                            db_word_metaphone_eng+=METAPHONE_ENG.encode(w)
                        w_ratio_metaphone_en=fuzz.WRatio(target_word_metaphone_eng,db_word_metaphone_eng)
                else:
                    if target_word_is_cyrillic:
                        w_ratio_orig1=rapidfuzz.string_metric.normalized_levenshtein(translit(target_word,"ru",reversed=True),db_word,weights=(2,2,1))
                        w_ratio_orig2=rapidfuzz.string_metric.normalized_levenshtein(target_word,translit(db_word,"ru"),weights=(2,2,1))
                        w_ratio_orig=max(w_ratio_orig1,w_ratio_orig2)

                        target_word_metaphone_ru = METAPHONE_RU.transform(target_word)
                        db_word_metaphone_ru = METAPHONE_RU.transform(translit(db_word,"ru"))
                        w_ratio_metaphone_ru=fuzz.WRatio(target_word_metaphone_ru,db_word_metaphone_ru)

                    else:
                        w_ratio_orig1=rapidfuzz.string_metric.normalized_levenshtein(translit(target_word,"ru"),db_word,weights=(2,2,1))
                        w_ratio_orig2=rapidfuzz.string_metric.normalized_levenshtein(target_word,translit(db_word,"ru",reversed=True),weights=(2,2,1))
                        w_ratio_orig=max(w_ratio_orig1,w_ratio_orig2)

                        target_word_arr=target_word.strip().split()
                        target_word_metaphone_eng=""
                        for w in target_word_arr:
                            target_word_metaphone_eng+=METAPHONE_ENG.encode(w)

                        db_word_arr=translit(target_word,"ru",reversed=True).strip().split()
                        db_word_metaphone_eng=""
                        for w in db_word_arr:
                            db_word_metaphone_eng+=METAPHONE_ENG.encode(w)
                        w_ratio_metaphone_en=fuzz.WRatio(target_word_metaphone_eng,db_word_metaphone_eng)
                
                if w_ratio_orig>70:
                    similar.add(key)
                # print(data)
        
            # if len(data[1][0])>=4:
            #     # rt=process.fuzz.token_set_ratio(query_text,data[1][0],force_ascii=False)
            #     if has_cyrillic(data[1][0]):
            #         temp_metaph=METAPHONE_RU.transform(data[1][0])
            #         dist=distance.levenshtein(metaphone_ru,temp_metaph)
            #         if dist<=1:
            #             print()
            #             print(dist,data[1][0])
            #             print(key,value)
            #             print()
            #     else:
            #         continue
            #         temp_metaph=METAPHONE_ENG.encode(data[1][0])
            #         dist=distance.levenshtein(metaphone_eng,temp_metaph)
            #         if dist<=1:
            #             print()
            #             print(dist,data[1][0])
            #             print(key,value)
            #             print()
    print(similar)
app = FastAPI()
@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/calculate_text_features")
async def calculate_text_features_handler(image: bytes = File(...), image_id: str = Form(...)):
    ocr_text_arr = get_ocr_text(image)
    if len(ocr_text_arr) == 0:
        return {"error":"not text detected"}
    IMG_ID_TXT_ARR[image_id]=ocr_text_arr
    add_descriptor(int(image_id),json.dumps(ocr_text_arr,ensure_ascii=False,cls=MyEncoder))
    return Response(status_code=status.HTTP_200_OK)


class Item_image_id(BaseModel):
    image_id: int

@app.post("/text_get_similar_images_by_id")
async def text_get_similar_images_by_id_handler(item: Item_image_id):
    try:
        ocr_text_arr = IMG_ID_TXT_ARR[item.image_id]
        similar = text_find_similar(ocr_text_arr)
        return similar
    except:
        raise HTTPException(
            status_code=500, detail="Image with this id is not found")


@app.post("/text_get_similar_images_by_image_buffer")
async def text_get_similar_images_by_image_buffer_handler(image: bytes = File(...)):
    ocr_text_arr = get_ocr_text(image)
    if len(ocr_text_arr) == 0:
        return {"error":"not text detected"}
    similar = text_find_similar(ocr_text_arr)
    return similar


@app.post("/delete_text_features")
async def delete_hist_features_handler(item: Item_image_id):
    try:
        del IMG_ID_TXT_ARR[item.image_id]
        delete_descriptor_by_id(item.image_id)
    except:
        pass
    return Response(status_code=status.HTTP_200_OK)

print(__name__)

if __name__ == 'text_web':
    create_table()
    # sync_db()
    init_index()
