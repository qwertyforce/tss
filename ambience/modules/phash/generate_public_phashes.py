from scipy.fft import dct
import cv2
import numpy as np
from numba import jit
from os import listdir
from joblib import Parallel, delayed
import sqlite3
import io
conn = sqlite3.connect('phashes.db')
IMAGE_PATH = "./../../../test"


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
    file_names = listdir(IMAGE_PATH)
    ids_in_db = set(get_all_ids())

    for file_name in file_names:
        file_id = int(file_name[:file_name.index('.')])
        if file_id in ids_in_db:
            ids_in_db.remove(file_id)
    for id in ids_in_db:
        delete_descriptor_by_id(id)  # Fix this
        print(f"deleting {id}")


@jit(cache=True, nopython=True)
def diff(dct, hash_size):
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff.flatten()


def fast_phash(image, hash_size=24, highfreq_factor=4): #hash_size=16 for 256bit #hash_size=8 for 64bit
    img_size = hash_size * highfreq_factor
    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)  # cv2.INTER_AREA
    dct_data = dct(dct(image, axis=0), axis=1)
    return diff(dct_data, hash_size)


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


def get_phash(query_image):
    bit_list_576 = fast_phash(query_image)
    phash = bit_list_to_72_uint8(bit_list_576)
    return phash

def calc_phash(file_name):
    file_id = int(file_name[:file_name.index('.')])
    img_path = IMAGE_PATH+"/"+file_name
    try:
        query_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        phash = get_phash(query_image)
    except:
        print("error")
        print(file_name)
        return None
    phash_bin=adapt_array(phash)
    return (file_id, phash_bin)


file_names = listdir(IMAGE_PATH)
print(len(file_names))
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
    phashes = Parallel(n_jobs=-1, verbose=1)(delayed(calc_phash)(file_name) for file_name in batch)
    phashes = [i for i in phashes if i]  # remove None's
    print("pushing data to db")
    conn.executemany('''INSERT INTO phashes(id, phash) VALUES (?,?)''', phashes)
    conn.commit()
