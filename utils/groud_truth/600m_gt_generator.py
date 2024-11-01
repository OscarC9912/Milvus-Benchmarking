import numpy as np
import faiss
import os
from tqdm import tqdm

def _split_file_reader(base_path):
    file_names = os.listdir(base_path)
    output = []
    for name in file_names:
        if name[0] == 'e':
            output.append(name)
    output.sort()
    return output


def faiss_cpu_l2_brute_force(base_path, query_path, gt_path, dist_path, dim, topk):

    # Load datasets
    file_names = _split_file_reader(base_path)
    base_data = np.concatenate([np.load(os.path.join(base_path, file)) for file in tqdm(file_names)], axis=0)
    query_data = np.load(query_path)[0:10]

    print(f"Base Data: {base_data.shape}")
    print(f"Query Data: {query_data.shape}")

    # Ensure the dimensions are as expected
    assert base_data.shape[1] == dim, "Base dataset dimension mismatch"
    assert query_data.shape[1] == dim, "Query dataset dimension mismatch"

    # Build the index
    index = faiss.IndexFlatL2(dim)

    # Add vectors to the index
    index.add(base_data)

    # Perform the search
    D, I = index.search(query_data, topk)

    # Save the results
    np.save(gt_path, I)
    np.save(dist_path, D)

    print(f"Success GT: {I.shape}")
    print(f"Success Distance: {D.shape}")
    
    
if __name__ == '__main__':
    base_path = '/ssddata/vecDB_publi_data/0.6b_128d_dataset'
    query_path = '/ssddata/vecDB_publi_data/src_zchenhj/0.6b_10k_query.npy'
    gt_path = '/ssddata/vecDB_publi_data/src_zchenhj/0.6b_10k_index.npy'
    dist_path = '/ssddata/vecDB_publi_data/src_zchenhj/0.6b_10k_dist.npy'
    dim, topk = 128, 100
    faiss_cpu_l2_brute_force(base_path, query_path, gt_path, dist_path, dim, topk)