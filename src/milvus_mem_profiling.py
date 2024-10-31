import time
import numpy as np
import json
from tqdm import tqdm
from memory_profiler import profile

from pymilvus import (
    connections,
    utility,BulkInsertState,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

config_index_param = {
    "FLAT": {},
    "IVF_FLAT": {'nlist': 100},
    "GPU_IVF_FLAT": {'nlist': 128},
    "IVF_SQ8": {'nlist': 128},
    "IVF_PQ": {'nlist': 256, 'm': 256},
    "SCANN": {'nlist': 150, 'with_raw_data': True},
    "GPU_IVF_PQ": {'nlist': 128, 'm': 8},
    "HNSW": {'M': 24, 'efConstruction': 80}
}


config_search_param = {
    "FLAT": {},
    "IVF_FLAT": {'nprobe': 32},
    "GPU_IVF_FLAT": {'nprobe': 8},
    "IVF_SQ8": {'nprobe': 120},
    "IVF_PQ": {'nprobe': 32},
    "SCANN": {'nprobe': 150, 'reorder_k': 150},
    "GPU_IVF_PQ": {'nprobe': 16},
    "HNSW": {'ef': 300}
}

class Milvus:
    
    def __init__(self):

        connections.connect("default", host="localhost", port="19530")
        
        self.collection_name = None
        self.collection = None
        
        self.data_name = None
        self.data = None
        
        self.num_entities = None
        self.dimension = None
        self.vectors_to_search = None
    
    def create_collection(self, data_name: str, collection_name: str, dimension: int, description: str):
        
        self.collection_name = collection_name
        
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name=data_name, dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]

        self.data_name = data_name
        schema = CollectionSchema(fields, description)
        
        self.collection = Collection(collection_name, schema, consistency_level="Strong")
        
        
    def dataloader(self, dataset_path, query_path):
        self.data = np.load(dataset_path)
        self.num_entities = self.data.shape[0]
        self.dimension = self.data.shape[1]
        self.vectors_to_search = np.load(query_path)[0:2000]
    
    @profile
    def insert_entitiy(self, batch_size=0):
        
        assert self.collection is not None
        
        def _batch_insert(collection, entities, batch_size):
            num_entities = len(entities[0])
            for i in tqdm(range(0, int(num_entities / batch_size) + 1)):    
                start_idx = i * batch_size
                end_index = min((i + 1) * batch_size, num_entities)
                if start_idx == end_index:
                    break
                batch = [entities[0][start_idx:end_index], entities[1][start_idx:end_index]]
                collection.insert(batch)

        entities = [
            [int(i) for i in range(self.num_entities)], 
            self.data
        ]
        
        if batch_size == 0:
            _ = self.collection.insert(entities)
        else:
            start_insert_time = time.time()
            _batch_insert(self.collection, entities, batch_size)
            end_insert_time = time.time()
            print(f"Insertion Time: {end_insert_time - start_insert_time}")
            self.collection.flush()
            print(f"Number of entities in Milvus: {self.collection.num_entities}")
            
    
    @profile
    def create_index(self, indexType: str):
        
        assert indexType in config_index_param
        self.indexType = indexType
        
        index = {
            "index_type": indexType,
            "metric_type": "L2",
            "params": config_index_param[indexType],
        }
        self.collection.drop_index()
        start_create_index = time.time()
        self.collection.create_index(self.data_name, index)
        end_create_index = time.time()
        self.time_create_index = end_create_index - start_create_index
        print(f"Index Time: {end_create_index - start_create_index}")
        
    
    @profile
    def topk_anns(self, top_k: int, metricType: str):
        
        self.collection.load()
        self.top_k = top_k
        self.metric = metricType
        
        search_params = {
            "metric_type": metricType,
            "params": config_search_param[self.indexType],
        }
        
        query_start_time = time.time()
        result = self.collection.search(self.vectors_to_search, self.data_name, search_params, limit=top_k)
        query_end_time = time.time()
        self.time_query = query_end_time - query_start_time
        print(f"Query Time: {self.time_query}")
        
        self.anns_result = []
        for hits in result:
            curr_result = set() # top-k neighbour of the current query point
            for hit in hits:
                curr_result.add(int(hit.id))
            self.anns_result.append(curr_result)
            
    def ground_truth_loader(self, gt_path: str):
        
        gt_data = np.load(gt_path)
        self.ground_truth = []
        
        for i in range(len(gt_data)):
            curr_id = set(gt_data[i].tolist())
            gts = {int(item) for item in curr_id}
            self.ground_truth.append(gts)
        
    def recall_calculator(self):
        assert len(self.anns_result) == len(self.ground_truth)        
        recalls = np.zeros(self.vectors_to_search.shape[0])
        for i in range(len(self.ground_truth)):
            currANN = self.anns_result[i]
            currGT = self.ground_truth[i]
            intersec = currANN.intersection(currGT)
            currRecall = len(intersec) / len(currGT) if currGT else 0
            recalls[i] = currRecall
            
        self.recall = np.mean(recalls)
        print(f"Recall: {self.recall}")


if __name__ == '__main__':
    
    # base_data_path = '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_base_medium.npy'
    # query_data_path = '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_query_large.npy'
    base_data_path = '/ssddata/vecDB_publi_data/10m_dataset/bert-nli-mean_base.npy'
    query_data_path = '/ssddata/vecDB_publi_data/10m_dataset/bert-nli-mean_query.npy'

    milvus = Milvus()
    milvus.create_collection('tes4t', 'tes4t', 768, "here is the experiment for milvus")
    milvus.dataloader(base_data_path, query_data_path)
    
    print('Profiling: Insertion')
    milvus.insert_entitiy(20000)
    
    print('Profiling: Index Creation')
    milvus.create_index("HNSW")
    # milvus.create_index("HNSW")
    
    print('Profiling: ANN Search')
    milvus.topk_anns(top_k=100, metricType="L2")