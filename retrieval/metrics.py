import numpy as np
from collections import defaultdict

def calculate_map(queries: list[dict], top_k_results: list[dict], verbose: bool = False):
    """
    Calculate mean Average Precision (mAP) for retrieval results

    Args:
        queries: List of dicts [{"name": str, "label": int/str}] - ground truth queries
        top_k_results: List of lists of dicts [[{"name": str, "label": int/str}, ...]] - retrieved items
                       for each query (must be same length as queries)
        verbose: Whether to print per-query AP scores

    Returns:
        mAP: float (mean average precision)
        detailed_results: List of dicts with per-query metrics
    """
    # Input validation
    if len(queries) != len(top_k_results):
        raise ValueError("Number of queries must match number of result sets")

    # Precompute relevant items per query
    # query_labels = [q['label'] for q in queries]
    # label_to_items = defaultdict(list)
    # for i, label in enumerate(query_labels):
    #     label_to_items[label].append(i)

    aps = []
    detailed_results = []

    for query_idx, (query, results) in enumerate(zip(queries, top_k_results)):
        query_label = query['label']
        # relevant_indices = label_to_items[query_label]

        # Calculate precision@k and recall@k at each position
        precision_at_k = []
        # recall_at_k = []
        relevant_count = 0

        for k, item in enumerate(results, 1):
            if item['label'] == query_label:
                relevant_count += 1
            precision = relevant_count / k
            # recall = relevant_count / len(relevant_indices)
            precision_at_k.append(precision)
            # recall_at_k.append(recall)

        # Calculate Average Precision (AP) for this query
        if precision_at_k:
            ap = np.mean(precision_at_k)
        else:
            ap = 0.0

        aps.append(ap)

        # Store detailed results
        # detailed = {
        #     'query_name': query['name'],
        #     'query_label': query_label,
        #     'ap': ap,
        #     'precision_at_k': precision_at_k,
        #     # 'recall_at_k': recall_at_k,
        #     'num_retrieved': len(results),
        #     'relevant': relevant_count
        # }
        detailed = {
            'query_name': query['name'],
            'query_label': query_label,
            'ap': ap,
            'relevant': relevant_count
        }
        detailed_results.append(detailed)

        if verbose:
            print(f"Query '{query['name']}' (label={query_label}): AP = {ap:.4f}")

    # Calculate mean Average Precision
    map_score = np.mean(aps)

    return map_score, detailed_results


# Example usage:
# if __name__ == "__main__":

#     queries = [
#         {"name": "query_1", "label": 0},
#     ]

#     # Retrieved items for each query (top_k=7)
#     top_k_results = [
#         [
#             {"name": "top_1", "label": 1},
#             {"name": "top_2", "label": 1},
#             {"name": "top_3", "label": 1},
#             {"name": "top_4", "label": 1},
#             {"name": "top_5", "label": 0},
#             {"name": "top_6", "label": 1},
#             {"name": "top_7", "label": 1},
#         ],
#     ]

#     # Calculate mAP
#     map_score, detailed = calculate_map(queries, top_k_results, verbose=True)

#     print(f"mAP: {map_score:.4f}")
