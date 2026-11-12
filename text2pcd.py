import numpy as np
from collections import defaultdict

def cosine_similarity(A, B):
    """
    Computes the cosine similarity between two matrices A and B.

    Args:
        A (np.ndarray): The first matrix of features (N, C).
        B (np.ndarray): The second matrix of features (M, C).

    Returns:
        np.ndarray: An (N, M) similarity matrix where sims[i, j] is the
                    cosine similarity between A[i, :] and B[j, :].
    """
    # Normalize the feature vectors to unit length
    # Adding a small epsilon to avoid division by zero for zero vectors, though
    # feature vectors are typically non-zero in these contexts.
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

    # Compute the dot product, which is the cosine similarity for normalized vectors
    sims = np.dot(A_norm, B_norm.T)
    return sims

def ndcg(pos_i, topk_, k):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG) for a set of queries.
    This implementation assumes binary relevance (1 for relevant, 0 for non-relevant).

    Args:
        pos_i (np.ndarray): An (npts, k) array where pos_i[i, 0] = 1.0 indicates
                            the ideal scenario where the correct item is at rank 0
                            for query `i`.
        topk_ (np.ndarray): An (npts, k) array where topk_[i, actual_rank] = 1.0
                            indicates that the correct item for query `i` was found
                            at `actual_rank`.
        k (int): The number of top results to consider for NDCG calculation.

    Returns:
        float: The mean NDCG score across all queries, scaled to 100.0.
    """
    if k == 0:
        return 0.0

    # Ranks are 0-indexed in the arrays (0 to k-1).
    # The discount factor for a relevant item at rank `r` is 1 / log2(r + 2).
    # For example, rank 0 (position 1) has discount 1/log2(2) = 1.
    # Rank 1 (position 2) has discount 1/log2(3).
    discounts = 1.0 / np.log2(np.arange(k) + 2)

    # Calculate Discounted Cumulative Gain (DCG) for each query
    # `topk_` contains 1s at the actual ranks where the correct item was found.
    dcg = np.sum(topk_ * discounts, axis=1)

    # Calculate Ideal Discounted Cumulative Gain (IDCG) for each query
    # `pos_i` contains 1s at rank 0, representing the ideal scenario (correct item always at rank 0).
    idcg = np.sum(pos_i * discounts, axis=1)

    # Avoid division by zero: if IDCG is 0 (e.g., no ideal relevant items, or k=0),
    # NDCG for that query is 0.
    ndcg_per_query = np.where(idcg > 0, dcg / idcg, 0)

    return np.mean(ndcg_per_query)

def text_to_pcd(txt_feat, pcd_feat, captions_list=None, return_ranks=False):
    """
    Performs Text to Point Cloud (Text Search) retrieval and evaluates metrics.
    For each text query, it searches for the most similar point cloud feature.

    Args:
        txt_feat (np.ndarray): An (N, C) matrix of text features, where N is
                               the number of text descriptions and C is the feature dimension.
                               Assumes a 1-to-1 correspondence with pcd_feat.
        pcd_feat (np.ndarray): An (N, C) matrix of point cloud features, where N is
                               the number of point clouds and C is the feature dimension.
        captions_list (list of str): A list of N captions, where captions_list[i]
                                     corresponds to txt_feat[i] and pcd_feat[i].
                                     Used for populating `ranks_dic`.
        return_ranks (bool): If True, the function will also return `ranks_dic`
                             containing detailed rank information.

    Returns:
        tuple: A tuple containing the retrieval metrics:
               (r1, r5, r10, medr, meanr, ndcg_val)
               If `return_ranks` is True, it also returns `ranks_dic`:
               (r1, r5, r10, medr, meanr, ndcg_val, ranks_dic)

    Metrics Explained:
        r1 (Recall@1): The percentage of text queries for which the correct
                       matching point cloud was found in the top 1 retrieved result.
                       A higher value indicates better performance.
        r5 (Recall@5): The percentage of text queries for which the correct
                       matching point cloud was found in the top 5 retrieved results.
                       A higher value indicates better performance.
        r10 (Recall@10): The percentage of text queries for which the correct
                        matching point cloud was found in the top 10 retrieved results.
                        A higher value indicates better performance.
        medr (Median Rank): The median rank (position, 1-indexed) of the correct
                            point cloud among all retrieved results for all text queries.
                            A lower value indicates better performance.
        meanr (Mean Rank): The average rank (position, 1-indexed) of the correct
                           point cloud among all retrieved results for all text queries.
                           A lower value indicates better performance.
        ndcg_val (Normalized Discounted Cumulative Gain): A measure of ranking quality
                                                      that considers the position of
                                                      relevant items. It assigns
                                                      higher scores to relevant items
                                                      appearing higher in the search
                                                      results. A higher value indicates
                                                      better ranking.
        ranks_dic (dict): (Returned only if `return_ranks` is True)
                          A dictionary where keys are the ground truth indices of
                          the text queries (0 to N-1). Each value is a list of tuples.
                          Each tuple contains:
                          - The original caption string for that query.
                          - A list of the top K (here, K=5) retrieved point cloud indices
                            for that query, sorted by similarity.
    """
    # N is the number of text descriptions / point clouds (assuming 1-to-1 correspondence)
    npts = txt_feat.shape[0]

    # Compute cosine similarity matrix where rows correspond to text queries
    # and columns correspond to point cloud candidates.
    # `sims[i, j]` represents the similarity between `txt_feat[i]` and `pcd_feat[j]`.
    # Shape of `sims` will be (N_txt, N_pcd) which is (N, N) in this 1-to-1 case.
    sims = cosine_similarity(txt_feat, pcd_feat)

    # Initialize arrays to store ranks and top-k hits
    ranks = np.zeros(npts, dtype=int) # Stores the rank of the correct match for each query

    # k is the number of top results to consider for R@K and NDCG
    k = 5
    topk_hit = np.zeros((npts, k)) # Binary: 1 if correct item is within top-k at its rank
    pos_i_ideal = np.zeros((npts, k)) # Binary: 1 at rank 0 for ideal scenario (used for NDCG IDCG)

    # For ideal NDCG, the relevant item is always at rank 0 (position 1)
    pos_i_ideal[:, 0] = 1.0

    ranks_dic = defaultdict(list) # Dictionary to store detailed rank information if requested

    # Iterate through each text query to find its matching point cloud
    for text_query_idx in range(npts):
        # `sims[text_query_idx, :]` gives the similarities of the current text query
        # (txt_feat[text_query_idx]) with ALL point cloud features (pcd_feat[0]...pcd_feat[N-1]).
        # `inds` will contain the indices of `pcd_feat` sorted in descending order of similarity
        # to `txt_feat[text_query_idx]`.
        inds = np.argsort(sims[text_query_idx, :])[::-1]

        # Find the rank (0-indexed) of the ground truth matching point cloud.
        # The ground truth match for `txt_feat[text_query_idx]` is `pcd_feat[text_query_idx]`.
        rank = np.where(inds == text_query_idx)[0][0]
        ranks[text_query_idx] = rank

        # Mark if the correct item was found within the top-k results at its specific rank
        if rank < k:
            topk_hit[text_query_idx, int(rank)] = 1.0

        if return_ranks:
            # Get the original caption for the current text query
            caption = captions_list[text_query_idx]
            # Get the indices of the top K retrieved point clouds
            top_pcd_indices = inds[:k].tolist()
            # Store the caption and the top K retrieved PCD indices
            ranks_dic[text_query_idx].append((caption, top_pcd_indices))

    # --- Compute Retrieval Metrics ---

    # Recall@1, Recall@5, Recall@10 (percentage of queries where correct item is within top K)
    r1 = 100.0 * np.sum(ranks < 1) / npts
    r5 = 100.0 * np.sum(ranks < 5) / npts
    r10 = 100.0 * np.sum(ranks < 10) / npts

    # Median Rank (MedR): The median position of the correct item in the ranked list.
    # Ranks are 0-indexed, so add 1 for 1-indexed interpretation.
    medr = np.floor(np.median(ranks)) + 1

    # Mean Rank (MeanR): The average position of the correct item in the ranked list.
    # Ranks are 0-indexed, so add 1 for 1-indexed interpretation.
    meanr = np.mean(ranks) + 1

    # Normalized Discounted Cumulative Gain (NDCG)
    ndcg_val = 100.0 * ndcg(pos_i_ideal, topk_hit, k)

    # Return the metrics, and optionally the detailed ranks dictionary
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ndcg_val, ranks_dic)
    else:
        return (r1, r5, r10, medr, meanr, ndcg_val)

def load_cap_feature(cap_feat_path, split='test', feat_type='prompt_avg'):
    allcap_feat = np.load(cap_feat_path, allow_pickle=True)
    ntrain = len(allcap_feat['train'])

    cap_feat = []
    for obj in allcap_feat[split]:
        cap_feat.append(obj['caption_feat'][feat_type])
    cap_feat = np.concatenate(cap_feat)
    
    return cap_feat, ntrain


def load_pred_feat(np_path, ntrain=4571, split='test'):
    allpred_feat = np.load(np_path, allow_pickle=True).item()

    if split == 'test':
       pred_feat = allpred_feat['text_feat'][ntrain:]
    elif split == 'train':
        pred_feat = allpred_feat['text_feat'][:ntrain]
    else:
        pred_feat = allpred_feat['text_feat']
        
    return pred_feat     

def multitext_to_pcd(txt_feat, pcd_feat, txt_ids, pcd_ids, captions_list=None, return_ranks=False):
    """
    Performs Text to Point Cloud (Text Search) retrieval and evaluates metrics.
    For each text query, it searches for the most similar point cloud feature.

    Args:
        txt_feat (np.ndarray): An (N, C) matrix of text features, where N is
                               the number of text descriptions and C is the feature dimension.
        pcd_feat (np.ndarray): An (n, C) matrix of point cloud features, where n is
                               the number of unique point clouds and C is the feature dimension.
                               Assumes N >= n.
        txt_ids (np.ndarray or list): A list or array of length N, where txt_ids[i]
                                      is the identifier (e.g., an integer ID or string)
                                      of the point cloud that txt_feat[i] describes.
        pcd_ids (np.ndarray or list): A list or array of length n, where pcd_ids[j]
                                      is the identifier of pcd_feat[j].
                                      txt_ids[i] == pcd_ids[j] implies txt_feat[i] describes pcd_feat[j].
        captions_list (list of str): A list of N captions, where captions_list[i]
                                     corresponds to txt_feat[i]. Used for populating `ranks_dic`.
        return_ranks (bool): If True, the function will also return `ranks_dic`
                             containing detailed rank information.

    Returns:
        tuple: A tuple containing the retrieval metrics:
               (r1, r5, r10, medr, meanr, ndcg_val)
               If `return_ranks` is True, it also returns `ranks_dic`:
               (r1, r5, r10, medr, meanr, ndcg_val, ranks_dic)

    Metrics Explained:
        r1 (Recall@1): The percentage of text queries for which the correct
                       matching point cloud was found in the top 1 retrieved result.
                       A higher value indicates better performance.
        r5 (Recall@5): The percentage of text queries for which the correct
                       matching point cloud was found in the top 5 retrieved results.
                       A higher value indicates better performance.
        r10 (Recall@10): The percentage of text queries for which the correct
                        matching point cloud was found in the top 10 retrieved results.
                        A higher value indicates better performance.
        medr (Median Rank): The median rank (position, 1-indexed) of the correct
                            point cloud among all retrieved results for all text queries.
                            A lower value indicates better performance.
        meanr (Mean Rank): The average rank (position, 1-indexed) of the correct
                           point cloud among all retrieved results for all text queries.
                           A lower value indicates better performance.
        ndcg_val (Normalized Discounted Cumulative Gain): A measure of ranking quality
                                                      that considers the position of
                                                      relevant items. It assigns
                                                      higher scores to relevant items
                                                      appearing higher in the search
                                                      results. A higher value indicates
                                                      better ranking.
        ranks_dic (dict): (Returned only if `return_ranks` is True)
                          A dictionary where keys are the ground truth indices of
                          the text queries (0 to N-1). Each value is a list of tuples.
                          Each tuple contains:
                          - The original caption string for that query.
                          - A list of the top K (here, K=5) retrieved point cloud IDs
                            for that query, sorted by similarity.
    """
    # N is the number of text descriptions
    num_txt_queries = txt_feat.shape[0]
    # n is the number of unique point clouds
    num_pcd_candidates = pcd_feat.shape[0]

    # Map pcd_ids to their indices for quick lookup
    pcd_id_to_idx = {pcd_id: idx for idx, pcd_id in enumerate(pcd_ids)}

    # Compute cosine similarity matrix where rows correspond to text queries
    # and columns correspond to point cloud candidates.
    # `sims[i, j]` represents the similarity between `txt_feat[i]` and `pcd_feat[j]`.
    # Shape of `sims` will be (N, n).
    sims = cosine_similarity(txt_feat, pcd_feat)

    # Initialize arrays to store ranks and top-k hits
    ranks = np.zeros(num_txt_queries, dtype=int) # Stores the rank of the correct match for each query

    # k is the number of top results to consider for R@K and NDCG
    k = 5
    # For NDCG, relevance is 1 for the ground truth Pcd and 0 for others.
    # ideal_dcg will be based on the ground truth item being at rank 0.
    # We will compute discounted cumulative gain for each query and then normalize.
    all_dcg = []
    all_idcg = []


    ranks_dic = defaultdict(list) # Dictionary to store detailed rank information if requested

    # Iterate through each text query to find its matching point cloud
    for text_query_idx in range(num_txt_queries):
        # Determine the ground truth point cloud ID for the current text query
        ground_truth_pcd_id = txt_ids[text_query_idx]

        # Find the index of the ground truth point cloud in pcd_feat
        # This is the target we are trying to retrieve
        ground_truth_pcd_feat_idx = pcd_id_to_idx[ground_truth_pcd_id]

        # `sims[text_query_idx, :]` gives the similarities of the current text query
        # (txt_feat[text_query_idx]) with ALL point cloud features (pcd_feat[0]...pcd_feat[n-1]).
        # `inds` will contain the indices of `pcd_feat` sorted in descending order of similarity
        # to `txt_feat[text_query_idx]`.
        inds = np.argsort(sims[text_query_idx, :])[::-1]

        # Find the rank (0-indexed) of the ground truth matching point cloud.
        rank = np.where(inds == ground_truth_pcd_feat_idx)[0][0]
        ranks[text_query_idx] = rank

        # Calculate for NDCG
        # Relevance score for retrieved items: 1 if it's the ground truth, 0 otherwise.
        # This assumes a binary relevance for simplicity, common in retrieval tasks
        # where the goal is to retrieve THE correct item.
        relevance_scores = np.zeros(num_pcd_candidates)
        relevance_scores[ground_truth_pcd_feat_idx] = 1.0

        # Create sorted relevance scores based on retrieved order for actual DCG
        retrieved_relevance = relevance_scores[inds]

        # Ideal relevance scores (ground truth item at rank 0) for IDCG
        ideal_relevance = np.zeros(num_pcd_candidates)
        ideal_relevance[0] = 1.0 # The ground truth item would ideally be at the top

        # Calculate DCG and IDCG for the current query
        # We use a simplified ndcg calculation for single relevant item here,
        # or rely on a proper ndcg_score function.
        # For a single relevant item at rank `r`, DCG = 1 / log2(r + 2)
        # For IDCG, relevant item at rank 0, IDCG = 1 / log2(0 + 2) = 1
        # Let's use the sklearn's ndcg_score for robustness, ensuring it expects (truth, predicted)
        # truth here being ideal_relevance and predicted being retrieved_relevance
        # Note: ndcg_score from sklearn typically expects relevance for all possible items.
        # For a single relevant item, a simpler manual calculation or careful use is needed.
        #
        # A more straightforward way for a single correct item:
        # DCG_q = sum_{i=1 to k} (rel_i / log2(i+1)) where rel_i is 1 if correct at rank i-1, else 0
        # IDCG_q = 1 / log2(1+1) = 1 (if k >= 1)
        # So, NDCG_q = (1 / log2(rank + 2)) / 1 if rank < k, else 0.
        # And we average these per query.

        if rank < k:
            all_dcg.append(1.0 / np.log2(rank + 2))
            all_idcg.append(1.0 / np.log2(0 + 2)) # Ideal: correct item at rank 0
        else:
            all_dcg.append(0.0) # Correct item not in top K, so DCG for top K is 0
            all_idcg.append(1.0 / np.log2(0 + 2))


        if return_ranks:
            # Get the original caption for the current text query
            caption = captions_list[text_query_idx]
            # Get the indices of the top K retrieved point clouds
            top_pcd_feature_indices = inds[:k].tolist()
            # Map these indices back to their original PCD IDs
            top_retrieved_pcd_ids = [pcd_ids[idx] for idx in top_pcd_feature_indices]
            # Store the caption and the top K retrieved PCD IDs
            ranks_dic[ground_truth_pcd_id].append([caption, top_retrieved_pcd_ids])


    # --- Compute Retrieval Metrics ---

    # Recall@1, Recall@5, Recall@10 (percentage of queries where correct item is within top K)
    r1 = 100.0 * np.sum(ranks < 1) / num_txt_queries
    r5 = 100.0 * np.sum(ranks < 5) / num_txt_queries
    r10 = 100.0 * np.sum(ranks < 10) / num_txt_queries

    # Median Rank (MedR): The median position of the correct item in the ranked list.
    # Ranks are 0-indexed, so add 1 for 1-indexed interpretation.
    medr = np.floor(np.median(ranks)) + 1

    # Mean Rank (MeanR): The average position of the correct item in the ranked list.
    # Ranks are 0-indexed, so add 1 for 1-indexed interpretation.
    meanr = np.mean(ranks) + 1

    # Normalized Discounted Cumulative Gain (NDCG)
    # Sum DCG and IDCG across all queries and then divide
    total_dcg = np.sum(all_dcg)
    total_idcg = np.sum(all_idcg)
    ndcg_val = 100.0 * (total_dcg / total_idcg) if total_idcg > 0 else 0.0


    # Return the metrics, and optionally the detailed ranks dictionary
    if return_ranks:
        return (r1, r5, r10, medr, meanr, ndcg_val, ranks_dic)
    else:
        return (r1, r5, r10, medr, meanr, ndcg_val)

