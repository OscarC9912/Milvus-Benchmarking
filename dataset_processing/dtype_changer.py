import os
import numpy as np
from tqdm import tqdm

parent_path = '/ssddata/folder2docker/milvus_folder/volumes/milvus/data/zchenhj/600m_dataset/segments'
folder_names = os.listdir(parent_path)
folder_names.sort()

for i in tqdm(range(len(folder_names))):
    curr_path = os.path.join(parent_path, folder_names[i])
    vector_path = os.path.join(curr_path, 'vector.npy')
    vec_data = np.load(vector_path)
    vec_data = vec_data.astype(np.float32)
    np.save(vector_path, vec_data)