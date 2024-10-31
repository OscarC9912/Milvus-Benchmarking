from tqdm import tqdm
import numpy as np
import faiss


def faiss_cpu_l1_brute_force(base_path, query_path, gt_path, dist_path, dim, topk):

    # Load datasets
    base_data = np.load(base_path).astype('float32')
    query_data = np.load(query_path).astype('float32')

    print(f"Base Data: {base_data.shape}")
    print(f"Query Data: {query_data.shape}")

    # Ensure the dimensions are as expected
    assert base_data.shape[1] == dim, "Base dataset dimension mismatch"
    assert query_data.shape[1] == dim, "Query dataset dimension mismatch"

    # # Build the index
    # index = faiss.IndexFlatL2(dim)
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_L1)

    # # Add vectors to the index
    # index.add(base_data)
    index.add(base_data)

    # # Perform the search
    D, I = index.search(query_data, topk)
    

    # Save the results
    np.save(gt_path, I)
    np.save(dist_path, D)

    print(f"Success GT: {I.shape}")
    print(f"Success Distance: {D.shape}")


def faiss_cpu_l2_brute_force(base_path, query_path, gt_path, dist_path, dim, topk):

    # Load datasets
    base_data = np.load(base_path).astype('float32')
    query_data = np.load(query_path).astype('float32')

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
    base = '/ssddata/vecDB_publi_data/10m_dataset/sentence-croissant-llm-base_base.npy'
    query = '/ssddata/vecDB_publi_data/10m_dataset/sentence-croissant-llm-base_query.npy'
    output_index = '/ssddata/vecDB_publi_data/10m_dataset/sentence-croissant-llm-base_base_index.npy'
    output_dist = '/ssddata/vecDB_publi_data/10m_dataset/sentence-croissant-llm-base_base_dist.npy'
    faiss_cpu_l2_brute_force(base, query, output_index, output_dist, 2048, 100)