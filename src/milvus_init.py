import time
import numpy as np
import json
from tqdm import tqdm
import math

from pymilvus import (
    connections,
    utility,BulkInsertState,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

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

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s', handlers=[
                        logging.FileHandler("/ssddata/zchenhj/vectorDB_study/bigdataset_test/0.6b_test.log"),  # Save logs to this file
                        logging.StreamHandler()  # Also print to console
                    ])

class Milvus:
    
    def __init__(self):

        connections.connect("default", host="localhost", port="19530")
        
        self.collection_name = None
        self.collection = None
        self.id_field = None
        
        self.data_name = None
        self.data = None
        
        self.num_entities = None
        self.dimension = None
        self.vectors_to_search = None
    
    
    def create_collection(self, data_name, collection_name, dimension, id_field='id', description="here is the experiment for milvus"):
        
        logging.info(f"Starting the New Collection")
        
        self.collection_name = collection_name
        self.id_field = id_field
        
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            
        fields = [
            FieldSchema(name=id_field, dtype=DataType.INT64, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name=data_name, dtype=DataType.FLOAT_VECTOR, dim=dimension)
        ]

        self.data_name = data_name
        schema = CollectionSchema(fields, description)
        
        self.collection = Collection(collection_name, schema, consistency_level="Strong")
        
        logging.info(f"Collection Name: {collection_name}")
        
              
    def retrieve(self, expr: str):
        result = self.collection.query(expr=expr, output_fields=[self.id_field])
        logging.info("query result with expression:", expr)
        for hit in result:
            logging.info(f"id: {hit['id_field']}")
        
        
    def dataloader(self, dataset_path, query_path, vec_search_size=None):
        
        self.data = np.load(dataset_path)
        self.num_entities = self.data.shape[0]
        self.dimension = self.data.shape[1]
        
        if vec_search_size is None:
            self.vectors_to_search = np.load(query_path)
        else:
            self.vectors_to_search = np.load(query_path)[0:vec_search_size]
        
        
    def insert_entitiy(self):
        
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
                
        def _batchsize_compute(max_size=64):
            batch_size = max_size * (1024**2) // (self.dimension * 4)
            nonint_part = int(str(batch_size)[1:])
            return int(batch_size - nonint_part)
            

        entities = [
            [int(i) for i in range(self.num_entities)], 
            self.data
        ]
        
        batch_size = _batchsize_compute(self.dimension)
        
        if batch_size == 0:
            _ = self.collection.insert(entities)
        else:
            start_insert_time = time.process_time() # time.time()
            _batch_insert(self.collection, entities, batch_size)
            end_insert_time = time.process_time() # time.time()
            logging.info(f"Insertion Time: {end_insert_time - start_insert_time} || Batchsize: {batch_size}")
            self.collection.flush()
            logging.info(f"Number of entities in Milvus: {self.collection.num_entities}")
            
            
    def create_index(self, indexType: str):
        
        assert indexType in config_index_param
        self.indexType = indexType
        
        task_index_param = dict()
        if indexType == 'IVF_FLAT':
            if self.num_entities <= 1000000:
                task_index_param['nlist'] = int(self.num_entities / 1000)
            else:
                task_index_param['nlist'] = int(math.sqrt(self.num_entities))
        else:
            task_index_param = config_index_param[indexType]
        
        index = {
            "index_type": indexType,
            "metric_type": "L2",
            "params": task_index_param
        }
        
        if self.collection.has_index():
            self.collection.drop_index()

        start_create_index = time.process_time() # time.time()
        self.collection.create_index(self.data_name, index)
        end_create_index = time.process_time() # time.time()
        
        self.time_create_index = end_create_index - start_create_index
        logging.info(f"Index: {indexType} | Parameter: {task_index_param} | Index Time (declare for bulk): {end_create_index - start_create_index}")
        
    
    def bulk_insert(self, file_paths):
        
        start_bulk_insert_time = time.process_time() # time.time()
        
        logging.info(f"Start Bulk Insert Workload ...")
        
        task_id = utility.do_bulk_insert(
            collection_name=self.collection_name,
            is_row_based=False,
            files=file_paths,
            using='default')
        
        task = utility.get_bulk_insert_state(task_id=task_id)
        
        logging.info(f"Bulk Insert TaskID: {task_id}")
        
        curr_task = None
        
        while 1:
            curr_task = utility.get_bulk_insert_state(task_id=task_id)
            if curr_task.state == BulkInsertState.ImportFailed:
                logging.info("Failed reason:", curr_task.failed_reason)
                break
            if curr_task.state == BulkInsertState.ImportCompleted:
                logging.info("Bulk Insertion Success")
                break    
        end_bulk_insert_time = time.process_time() #time.time()
        
        
        start_wait_index_time = time.process_time() #time.time()
        
        utility.wait_for_index_building_complete(self.collection_name)
        res = utility.index_building_progress(self.collection_name)
        index_completed = res["pending_index_rows"] == 0
        
        while not index_completed:
            res = utility.index_building_progress(self.collection_name)
            # logging.info(f"index building progress: {res}")
            index_completed = res["pending_index_rows"] == 0
        
        end_wait_index_time = time.process_time() #time.time()
        logging.info(f"index building progress: {res}")
        
        delta = end_bulk_insert_time - start_bulk_insert_time
        delta2 = end_wait_index_time - start_wait_index_time
        logging.info(f"Bulk Insert: {delta} & Indexing Time: {delta2}")
        
        
    def bulk_insert_splitted(self, file_paths):
        """
        file_paths: [[id.npy, vector.npy], ...]
        """
        
        start_bulk_insert_time = time.process_time() #time.time()
        
        logging.info(f"Start Bulk Insert Workload ...")
        
        taskid_collection = []
        for segement in file_paths:
            task_id = utility.do_bulk_insert(
                collection_name=self.collection_name,
                is_row_based=False,
                files=segement, using='default')
            taskid_collection.append(task_id)
            
        logging.info(f"Bulk Insert TaskID: {taskid_collection}")
        
        curr_task = None
        completed = []
        
        while len(completed) != len(taskid_collection):
            for task_id in taskid_collection:
                if task_id not in completed:
                    curr_task = utility.get_bulk_insert_state(task_id=task_id)
                    if curr_task.state == BulkInsertState.ImportFailed:
                        logging.info("Failed reason:", curr_task.failed_reason)
                        break
                    if curr_task.state == BulkInsertState.ImportCompleted:
                        logging.info(f"Task ID: {task_id} Completed")
                        completed.append(task_id)
                    
        end_bulk_insert_time = time.process_time()# time.time()
        
        
        start_wait_index_time = time.process_time() #time.time()
        
        utility.wait_for_index_building_complete(self.collection_name)
        res = utility.index_building_progress(self.collection_name)
        index_completed = res["pending_index_rows"] == 0
        
        while not index_completed:
            res = utility.index_building_progress(self.collection_name)
            # logging.info(f"index building progress: {res}")
            index_completed = res["pending_index_rows"] == 0
        
        end_wait_index_time = time.process_time() #time.time()
        logging.info(f"index building progress: {res}")
        
        delta = end_bulk_insert_time - start_bulk_insert_time
        delta2 = end_wait_index_time - start_wait_index_time
        logging.info(f"Bulk Insert: {delta} & Indexing Time: {delta2}")
        
        
    def test_bulk_insert(self, file_paths):
        
        start_bulk_insert_time = time.process_time() #time.time()
        
        task_id = utility.do_bulk_insert(
            collection_name=self.collection_name,
            is_row_based=False,
            files=file_paths,
            using='default')
        
        curr_task = None
        
        while 1:
            curr_task = utility.get_bulk_insert_state(task_id=task_id)
            if curr_task.state == BulkInsertState.ImportFailed:
                logging.info("Failed reason:", curr_task.failed_reason)
                break
            if curr_task.state == BulkInsertState.ImportCompleted:
                logging.info("Bulk Insertion Success")
                break    
        end_bulk_insert_time = time.process_time() #time.time()
        
        delta = end_bulk_insert_time - start_bulk_insert_time
        logging.info(f"Bulk Insert without Index: {delta}")
    
    
    def topk_anns(self, top_k: int, metricType: str):
        
        try:
            self.collection.load()
            self.collection.load(_refresh=True)
        except:
            pass
        
        self.top_k = top_k
        self.metric = metricType
        
        search_params = {
            "metric_type": metricType,
            "params": config_search_param[self.indexType],
        }
        
        query_start_time = time.process_time() #time.time()
        result = self.collection.search(data=self.vectors_to_search, 
                                        anns_field=self.data_name,
                                        param=search_params, limit=top_k,
                                        output_fields=[self.id_field])
        query_end_time = time.process_time() #time.time()
        
        self.time_query = query_end_time - query_start_time
        
        logging.info(f"Query Size: {len(self.vectors_to_search)} || Query Time: {self.time_query} || QPS: {len(self.vectors_to_search) / self.time_query}")
        
        self.anns_result = []
        for hits in result:
            curr_result = set() # top-k neighbour of the current query point
            for hit in hits:
                curr_result.add(int(hit.id))
            self.anns_result.append(curr_result)    
            
            
    def ground_truth_loader(self, gt_path: str):
        
        gt_data = np.load(gt_path)[0:len(self.vectors_to_search)]
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
        logging.info(f"Recall: {self.recall}")
        logging.info("Current Run Complete\n")