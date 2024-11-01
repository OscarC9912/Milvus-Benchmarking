from src.milvus_init import Milvus


if __name__ == '__main__':
    
    print('Executing the 768 Dim')
    
    bert_base_data = ['/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_base_small.npy',
                      '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_base_medium.npy',
                      '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_base_large.npy']
    
    bert_query_data = ['/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_query_large.npy'] * 3
    
    ground_truths = ['/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_gtid_small.npy',
                     '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_gtid_medium.npy',
                     '/ssddata/vecDB_publi_data/test_data_0919/wikisum_bert-nli-mean_gtid_large.npy']
    
    names = ['small', 'medium', 'large']
    
    for i in range(len(bert_query_data)):
        
        print(f"=== {names[i]} ===")
        
        for index in ["IVF_FLAT", "HNSW"]:
    
            milvus = Milvus()
            
            milvus.create_collection('tes22', 'tes22', 768, "here is the experiment for milvus")
            milvus.dataloader(bert_base_data[i], bert_query_data[i], vec_search_size=5000)
            milvus.insert_entitiy()
            milvus.create_index(index)
            milvus.topk_anns(top_k=100, metricType="L2")
            milvus.ground_truth_loader(ground_truths[i])
            milvus.recall_calculator()
            
    
    print('Executing the 2048 Dim')
            
    sc_base_data = ['/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_base_small.npy',
                      '/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_base_medium.npy',
                      '/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_base_large.npy']
    
    sc_query_data = ['/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_query_large.npy'] * 3
    
    ground_truths = ['/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_gtid_small.npy',
                     '/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_gtid_medium.npy',
                     '/ssddata/vecDB_publi_data/test_data_0919/wikisum_sentence-croissant-llm-base_gtid_large.npy']
    

    # names = ['small', 'medium', 'large']
    
    for i in range(len(sc_base_data)):
        
        print(f"=== {names[i]} ===")
        
        for index in ["IVF_FLAT"]:
            milvus = Milvus()

            milvus.create_collection('tes33t', 'tes33t', 2048, "here is the experiment for milvus")
            milvus.dataloader(sc_base_data[i], sc_query_data[i], vec_search_size=5000)
            milvus.insert_entitiy()
            milvus.create_index(index)
            milvus.topk_anns(top_k=100, metricType="L2")
            milvus.ground_truth_loader(ground_truths[i])
            milvus.recall_calculator()