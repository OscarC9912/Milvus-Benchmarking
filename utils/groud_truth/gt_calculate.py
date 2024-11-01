import numpy as np
import faiss
import os
from tqdm import tqdm

def _split_file_reader(base_path):
    assert 'segments' in base_path, "The base path should contain the 'segments' directory"
    file_names = os.listdir(base_path)
    file_names.sort()
    output = []
    for name in file_names:
        path_segment = os.path.join(base_path, name)
        data_name = os.path.join(path_segment, 'vector.npy')
        output.append(data_name)
    output.sort()
    return output


def faiss_cpu_l2_brute_force(base_path, query_path, gt_path, dist_path, dim, topk):

    # Load datasets
    file_names = _split_file_reader(base_path)
    base_data = np.concatenate([np.load(vfile) for vfile in tqdm(file_names)], axis=0)
    query_data = np.load(query_path)[0:100]

    print(f"Base Data: {base_data.shape}")
    print(f"Query Data: {query_data.shape}")

    # Ensure the dimensions are as expected
    assert base_data.shape[1] == dim, "Base dataset dimension mismatch"
    assert query_data.shape[1] == dim, "Query dataset dimension mismatch"

    index = faiss.IndexFlatL2(dim) # Build the index
    index.add(base_data) # Add vectors to the index
    D, I = index.search(query_data, topk) # Perform the search

    # Save the results
    np.save(gt_path, I)
    np.save(dist_path, D)

    print(f"Success GT: {I.shape}")
    print(f"Success Distance: {D.shape}")
    
    
if __name__ == '__main__':
    base_path = '/home/vecDB_publi_data/600m_dataset/segments'
    query_path = '/home/zchenhj/workspace/vectorDB_study/temp_dataset/0.6b_10k_query.npy'
    gt_path = '/home/zchenhj/workspace/vectorDB_study/ground_truth/0.6b_100_index.npy'
    dist_path = '/home/zchenhj/workspace/vectorDB_study/ground_truth/0.6b_100_dist.npy'
    dim, topk = 128, 100
    faiss_cpu_l2_brute_force(base_path, query_path, gt_path, dist_path, dim, topk)