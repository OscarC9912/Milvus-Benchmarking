import time
import numpy as np
import json
from tqdm import tqdm
import os
from milvus_init import Milvus


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


def pipeline_split_insert(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index_type, vec_search_size=None):

    milvus = Milvus()
    milvus.create_collection(collection_name, collection_name, dimension, id_field)

    milvus.vectors_to_search = np.load(query_path)
    if vec_search_size is not None:
        milvus.vectors_to_search = np.load(query_path)[0:vec_search_size]
    
    milvus.num_entities = num_entity
    milvus.dimension = dimension
    
    milvus.create_index(index_type)

    milvus.bulk_insert_splitted(data_paths)

    # milvus.ground_truth_loader(gt_path)

    milvus.topk_anns(top_k=100, metricType="L2")

    # milvus.recall_calculator()
    
    
def insert_test_pipeline(collection_name, id_field, num_entity, dimension, data_paths):

    milvus = Milvus()
    milvus.create_collection(collection_name, collection_name, dimension, id_field)
    
    milvus.num_entities = num_entity
    milvus.dimension = dimension
    
    milvus.test_bulk_insert(data_paths)
    
    

if __name__ == '__main__':
    
    # sizes = ['small', 'medium', 'large']
    # id_fields = ['id_field_100k', 'id_field_500k', 'id_field_1M']
    # num_entities = [100000, 500000, 1000000]
    
    # for curr_index in ['IVF_FLAT', 'HNSW']:
    
    #     for i in range(len(sizes)):
        
    #         collection_name = f"wikisum_bert_nli_mean_base_{sizes[i]}"
    #         id_field = id_fields[i]
    #         num_entity, dimension = num_entities[i], 768
            
    #         data_paths = [
    #             f'/var/lib/milvus/data/zchenhj/embedding_data/{id_fields[i]}.npy',
    #             f'/var/lib/milvus/data/zchenhj/embedding_data/wikisum_bert_nli_mean_base_{sizes[i]}.npy'
    #         ]
            
    #         query_path = '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_query_large.npy'
            
    #         gt_path = f'/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_gtid_{sizes[i]}.npy'
            
    #         index_type = curr_index
        
    #         pipeline(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index_type)
            
            
    # for curr_index in ['IVF_FLAT', 'HNSW']:
    
    #     for i in range(len(sizes)):
        
    #         collection_name = f"wikisum_sentence_croissant_llm_base_base_{sizes[i]}"
    #         id_field = id_fields[i]
    #         num_entity, dimension = num_entities[i], 2048
            
    #         data_paths = [
    #             f'/var/lib/milvus/data/zchenhj/embedding_data/{id_fields[i]}.npy',
    #             f'/var/lib/milvus/data/zchenhj/embedding_data/wikisum_sentence_croissant_llm_base_base_{sizes[i]}.npy'
    #         ]
            
    #         query_path = '/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_query_large.npy'
            
    #         gt_path = f'/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_gtid_{sizes[i]}.npy'
            
    #         index_type = curr_index
        
    #         pipeline(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index_type, 5000)
    
    # sizes = ['small', 'medium', 'large']
    # id_fields = ['id_field_100k', 'id_field_500k', 'id_field_1M']
    # num_entities = [100000, 500000, 1000000]
    
    
    # for i in range(len(sizes)):
    
    #     collection_name = f"wikisum_bert_nli_mean_base_{sizes[i]}"
    #     id_field = id_fields[i]
    #     num_entity, dimension = num_entities[i], 768
        
    #     data_paths = [
    #         f'/var/lib/milvus/data/zchenhj/embedding_data/{id_fields[i]}.npy',
    #         f'/var/lib/milvus/data/zchenhj/embedding_data/wikisum_bert_nli_mean_base_{sizes[i]}.npy'
    #     ]
        
    
    #     insert_test_pipeline(collection_name, id_field, num_entity, dimension, data_paths)
            
            
    # for curr_index in ['IVF_FLAT']:
    
    #     for i in range(len(sizes)):
        
    #         collection_name = f"wikisum_sentence_croissant_llm_base_base_{sizes[i]}"
    #         id_field = id_fields[i]
    #         num_entity, dimension = num_entities[i], 2048
            
    #         data_paths = [
    #             f'/var/lib/milvus/data/zchenhj/embedding_data/{id_fields[i]}.npy',
    #             f'/var/lib/milvus/data/zchenhj/embedding_data/wikisum_sentence_croissant_llm_base_base_{sizes[i]}.npy'
    #         ]
        
    #         insert_test_pipeline(collection_name, id_field, num_entity, dimension, data_paths)
    
    # collection_name = 'bert_nli_mean_base'
    # num_entity, dimension = 12641221, 768
    # id_field = 'id_12m'
    
    # data_paths = [
    #     f'/var/lib/milvus/data/zchenhj/12m_data/id_12m.npy',
    #     f'/var/lib/milvus/data/zchenhj/12m_data/bert_nli_mean_base.npy']
    
    # query_path = '/ssddata/vecDB_publi_data/10m_dataset/bert-nli-mean_query.npy'
    # gt_path = '/ssddata/vecDB_publi_data/10m_dataset/bert-nli-mean_base_index.npy'
    
    # for index in ["IVF_FLAT", "HNSW"]:
    #     pipeline(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index, vec_search_size=None)

        
    # collection_name = 'sentence_croissant_llm_base_base'
    # num_entity, dimension = 12641221, 2048
    # id_field = 'id_12m'
    
    # data_paths = [
    #     f'/var/lib/milvus/data/zchenhj/12m_data/id_12m.npy',
    #     f'/var/lib/milvus/data/zchenhj/12m_data/sentence_croissant_llm_base_base.npy']
    
    # query_path = '/ssddata/vecDB_publi_data/10m_dataset/sentence-croissant-llm-base_query.npy'
    # gt_path = '/ssddata/vecDB_publi_data/10m_dataset/sentence-croissant-llm-base_base_index.npy'
    
    # # for index in ["IVF_FLAT", "HNSW"]:
    # for index in ["HNSW"]:
    #     pipeline(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index, vec_search_size=500)
    
    collection_name = 'vector'
    num_entity, dimension = 597909919, 128
    id_field = 'id'
    
    parent_path = '/ssddata/folder2docker/milvus_folder/volumes/milvus/data/zchenhj/600m_dataset/segments'
    docker_path = '/var/lib/milvus/data/zchenhj/600m_dataset/segments'
    data_paths, subfolders = [], os.listdir(parent_path)
    for i in tqdm(range(len(subfolders))):
        curr_path = os.path.join(docker_path, subfolders[i])
        id_path, vec_path = os.path.join(curr_path, 'id.npy'), os.path.join(curr_path, 'vector.npy')
        data_paths.append([id_path, vec_path])
        
    print(f"{len(data_paths)} Samples")
    query_path = '/ssddata/vecDB_publi_data/src_zchenhj/0.6b_10k_query.npy'
    gt_path = ''
    
    # for index in ["IVF_FLAT", "HNSW"]:
    for index in ["HNSW"]:
        pipeline_split_insert(collection_name, id_field, num_entity, dimension, data_paths, query_path, gt_path, index)
        

    