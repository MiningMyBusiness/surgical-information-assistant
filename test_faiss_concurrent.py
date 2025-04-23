import multiprocessing
import time
from utils.index_w_faiss import FaissReader
import sys

def query_index(reader, query, process_id):
    start_time = time.time()
    results = reader.query(query)
    end_time = time.time()
    print(f"Process {process_id}: Query '{query}' took {end_time - start_time:.4f} seconds")
    print(f"Process {process_id}: Found {len(results)} results")
    return results

def run_concurrent_queries(index_path, queries, num_processes):
    reader = FaissReader(index_path)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        jobs = []
        for i, query in enumerate(queries):
            job = pool.apply_async(query_index, (reader, query, i))
            jobs.append(job)
        
        results = [job.get() for job in jobs]
    
    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        index_path = sys.argv[1]
    else:
        index_path = "surgical_faiss_index"
    queries = [
        "What is the treatment for appendicitis?",
        "How is a laparoscopic cholecystectomy performed?",
        "What are the complications of hernia repair?",
        "Describe the steps of a mastectomy procedure.",
        "What is the recovery process after knee replacement surgery?"
    ]
    num_processes = 4  # You can adjust this based on your system's capabilities

    print(f"Starting concurrent queries with {num_processes} processes...")
    start_time = time.time()
    results = run_concurrent_queries(index_path, queries, num_processes)
    end_time = time.time()

    print(f"\nAll queries completed in {end_time - start_time:.4f} seconds")
    print(f"Total results found: {sum(len(r) for r in results)}")