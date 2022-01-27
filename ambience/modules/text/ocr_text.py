import numpy as np
import cv2
from os import listdir
import sqlite3
import math
from paddleocr import PaddleOCR
import json
from tqdm import tqdm
conn = sqlite3.connect('ocr_text_ru_eng.db')
IMAGE_PATH = "./content/"

ocr_ru = PaddleOCR(use_angle_cls=True, lang="ru",show_log = False)
ocr_en = PaddleOCR(use_angle_cls=True, lang="en",show_log = False)

def resize_img_to_threshold(img):
    height,width=img.shape
    threshold=2000*1500
    if height*width>threshold:
        k=math.sqrt(height*width/(threshold))
        img=cv2.resize(img, (round(width/k),round(height/k)), interpolation=cv2.INTER_LINEAR)
    return img

def create_table():
	cursor = conn.cursor()
	query = '''
	    CREATE TABLE IF NOT EXISTS ocr_text(
	    	id TEXT NOT NULL UNIQUE PRIMARY KEY,
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

def sync_db():
    file_names = listdir(IMAGE_PATH)
    ids_in_db = set(get_all_ids())

    for file_name in file_names:
        if file_name in ids_in_db:
            ids_in_db.remove(file_name)
    for id in ids_in_db:
        delete_descriptor_by_id(id)  # Fix this
        print(f"deleting {id}")

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

def get_text(img_path):
    file_id = int(file_name[:file_name.index('.')])
    img_path = IMAGE_PATH+"/"+file_name
    try:
      img=cv2.imread(img_path,0)
      img=resize_img_to_threshold(img)
      img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[255,255,255])
    except:
      print("error")
      print(img_path)
      return None
    result_ru = ocr_ru.ocr(img, cls=True)
    result_en = ocr_en.ocr(img, cls=True)

    result_ru = [a for a in result_ru if a[1][1]>0.6]
    result_en = [a for a in result_en if a[1][1]>0.6]
    if len(result_en) + len(result_ru) == 0:
      return None

    final_words = []
    for txt in result_ru:
        coords_ru=str(txt[0])
        flag1 = False
        for _txt in result_en:
            coords_en=str(_txt[0])
            if coords_ru == coords_en:
                flag1 = True
                if txt[1][1] > _txt[1][1]:
                    final_words.append(txt)
                else:
                    final_words.append(_txt)
                break
        if flag1 == False:
            final_words.append(txt)

    for txt in result_en: 
        coords_en=str(txt[0])
        flag1 = False
        for _txt in final_words:
            coords=str(_txt[0])
            if coords_en == coords:
                flag1 = True
                break
        if flag1 == False:
            final_words.append(txt)

    return (file_id,  json.dumps(final_words,ensure_ascii=False,cls=MyEncoder))

file_names = listdir(IMAGE_PATH)
create_table()
# sync_db()
new_images = []
for file_name in file_names:
    file_id = int(file_name[:file_name.index('.')])
    if check_if_exists_by_id(file_id):
        continue
    new_images.append(file_name)
print(len(new_images))
new_images = [new_images[i:i + 100000] for i in range(0, len(new_images), 100000)]
for batch in new_images:
    ocr_text_batch=[]
    for file_name in tqdm(batch):
        words=get_text(file_name)
        if words:
            ocr_text_batch.append(words)
    print("pushing data to db")
    conn.executemany('''INSERT INTO ocr_text(id, text_arr) VALUES (?,?)''', ocr_text_batch)
    conn.commit()