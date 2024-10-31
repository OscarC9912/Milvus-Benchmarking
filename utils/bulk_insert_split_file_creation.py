import os
import time
import numpy as np
import json
from tqdm import tqdm
import subprocess
from src.milvus_init import Milvus


def pipeline(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index_type, vec_search_size=None):

    milvus = Milvus()
    milvus.create_collection(collection_name, collection_name, dimension, id_field)

    milvus.vectors_to_search = np.load(query_path)
    if vec_search_size is not None:
        milvus.vectors_to_search = np.load(query_path)[0:vec_search_size]
    
    milvus.num_entities = num_entity
    milvus.dimension = dimension
    
    milvus.create_index(index_type)
    milvus.bulk_insert(data_paths)
    milvus.ground_truth_loader(gt_path)
    milvus.topk_anns(top_k=100, metricType="L2")
    milvus.recall_calculator() 


def _split_file_reader(file_path, starting):
        file_names = os.listdir(file_path)
        output = []
        for name in file_names:
            if name[0] == starting:
                output.append(name)
        output.sort()
        # print(f"Here are all names: {output}")
        return output
    

def id_file_creator(file_path, starting, save_dir):

    file_names = _split_file_reader(file_path, starting)
    starting, ending = 0, None
    
    if file_path[-1] != '/':
        file_path = f"{file_path}/"
    if save_dir[-1] != '/':
        save_dir = f"{save_dir}/"
    
    for i, single_file in tqdm(enumerate(file_names)):
        
        curr_path = f"{file_path}{single_file}"
            
        curr_len = len(np.load(curr_path))
        ending = starting + curr_len
        
        ids = np.array([i for i in range(starting, ending)])
        
        assert len(ids) == curr_len
        
        curr_save_path = f"{save_dir}segments/segment{i}/"
        
        if not os.path.exists(curr_save_path):
            os.makedirs(curr_save_path)
            
        save_file_path = os.path.join(curr_save_path, "id.npy")
        np.save(save_file_path, ids)
        
        starting = ending
        
        
def copier(file_path, destination_path):

    if file_path[-1] != '/':    
        file_path = f"{file_path}/"
    if destination_path[-1] != '/':    
        destination_path = f"{destination_path}/"

    file_names = _split_file_reader(file_path, 'e')
    
    for i in tqdm(range(len(file_names))):
        source_file_name = f"{file_path}{file_names[i]}"
        destination_name = f"{destination_path}segment{i}/vector.npy"
        subprocess.run(['cp', source_file_name, destination_name], check=True)
        

            
if __name__ == '__main__':
    # file_path = '/ssddata/vecDB_publi_data/0.6b_128d_dataset'
    # save_dir = '/ssddata/vecDB_publi_data/'
    # id_file_creator(file_path, starting, save_dir)
    file_path = '/ssddata/vecDB_publi_data/0.6b_128d_dataset'
    destination_path = '/ssddata/folder2docker/milvus_folder/volumes/milvus/data/zchenhj/600m_dataset/segments'
    copier(file_path, destination_path)
        
        
        
         
        
        
        
    
    
    


    