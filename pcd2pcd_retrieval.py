import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

def compute_retrieval_mAP(database_feat, query_feat, database_class, query_class):
    """
    Compute Average Precision (AP) for each query, class-wise mAP, and overall mAP,
    only for the intersection of classes between the query and database sets.

    Args:
        database_feat (ndarray): (N, C) array of database feature vectors
        query_feat (ndarray): (Q, C) array of query feature vectors
        database_class (list of str): (N,) class labels of the database items
        query_class (list of str): (Q,) class labels of the query items

    Returns:
        aps (list of float): AP for each query
        class_map (dict): mean AP for each class
        mean_ap (float): overall mean AP
    """
    database_class = np.array(database_class)
    query_class = np.array(query_class)

    # Step 1: Identify common classes
    common_classes = set(database_class).intersection(set(query_class))
    print(f"Number of common classes: {len(common_classes)}")

    # Step 2: Filter database and query to include only those with common classes
    db_mask = np.isin(database_class, list(common_classes))
    q_mask = np.isin(query_class, list(common_classes))

    database_feat = database_feat[db_mask]
    query_feat = query_feat[q_mask]
    database_class = database_class[db_mask]
    query_class = query_class[q_mask]

    # Step 3: Compute cosine similarity
    similarity = 1 - cdist(query_feat, database_feat, metric='cosine')  # (Q, N)
    aps = []
    classwise_aps = defaultdict(list)

    for i in range(query_feat.shape[0]):
        query_label = query_class[i]
        sim_row = similarity[i]

        # Rank database by similarity
        sorted_idx = np.argsort(-sim_row)
        sorted_labels = database_class[sorted_idx]

        # Binary relevance vector
        relevant = (sorted_labels == query_label).astype(np.int32)
        GTP = np.sum(relevant)
        if GTP == 0:
            aps.append(0.0)
            continue

        cumsum = np.cumsum(relevant)
        precision_at_k = cumsum / (np.arange(len(relevant)) + 1)
        ap = np.sum(precision_at_k * relevant) / GTP

        aps.append(ap)
        classwise_aps[query_label].append(ap)

    # Class-wise mAP
    class_map = {cls: np.mean(ap_list) for cls, ap_list in classwise_aps.items()}

    # Overall mAP
    mean_ap = np.mean(aps)

    return aps, class_map, mean_ap

# OmniObject3d ------------------------------------------------------------------
omni3d_path = './data/OmniObject3D/omniobject3d_color.npz'
omni3d_data = np.load(omni3d_path, allow_pickle=True)
len_train = omni3d_data['train'].shape[0]

def compute_omni3d_retrieval_mAP(path, so3_path=None, model='tamm_pointbert'):
    data = np.load(path, allow_pickle=True).item()

    db_feat, qr_feat = data['features'][:len_train], data['features'][len_train:]
    db_cls, qr_cls = data['categories'][:len_train], data['categories'][len_train:]

    _, class_mAP, mAP = compute_retrieval_mAP(
        db_feat, qr_feat,
        db_cls, qr_cls
    )
    print(model, 'mAP (no rot-no rot):', mAP)
    
    if so3_path:
        so3_data = np.load(so3_path, allow_pickle=True).item()       
        db_so3_feat, qr_so3_feat = so3_data['features'][:len_train], so3_data['features'][len_train:]
        db_so3_cls, qr_so3_cls = so3_data['categories'][:len_train], so3_data['categories'][len_train:]
        _, class_mAP, mAP = compute_retrieval_mAP(
            db_feat, qr_so3_feat,
            db_cls, qr_so3_cls
        )
        print(model, 'mAP (no rot-So3):', mAP)

        _, class_mAP, mAP = compute_retrieval_mAP(
            db_so3_feat, qr_feat,
            db_so3_cls, qr_cls
        )
        print(model, 'mAP (so3-no rot):', mAP)

        _, class_mAP, mAP = compute_retrieval_mAP(
            db_so3_feat, qr_so3_feat,
            db_so3_cls, qr_so3_cls
        )
        print(model, 'mAP (so3-so3):', mAP)

# rimamba_path = './exp/eval/ri_mamba_color_retrieval_coloromni3d@20250529-122252/omniobject3d_featall.npy'
rimamba_path = './exp/eval/ri_mamba_cia_retrieval_ensemblelvis_fixedcoloromni3d@20250603-204831/omniobject3d_featall.npy'
# rimamba_so3_path = './exp/eval/ri_mamba_color_retrieval_so3_coloromni3d@20250529-122735/omniobject3d_featall_so3.npy'
compute_omni3d_retrieval_mAP(rimamba_path, None, 'rimamba_nocia')

tamm_path = './exp/eval/pointbert_retrieval_coloromni3d@20250529-123026/omniobject3d_featall.npy'
tamm_so3_path = './exp/eval/pointbert_retrieval_so3_coloromni3d@20250529-123258/omniobject3d_featall_so3.npy'
compute_omni3d_retrieval_mAP(tamm_path, tamm_so3_path) 
    
# TAMM-PointBERT - no rotation (shapenet)
shapenet_data = np.load('./exp/eval/pointbert@20250521-204815/shapenet_feat.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/pointbert@20250521-204815/modelnet40_feat.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
# TAMM-PointBERT - no rotation (ensemble)
shapenet_data = np.load('./exp/eval/pointbert_retrieval_ensemble@20250522-104320/shapenet_feat.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/pointbert_retrieval_ensemble@20250522-104320/modelnet40_feat.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print('mAP no rot (ensemble) tamm:', mAP)

# TAMM-PointBERT - no rotation/So3
shapenet_data = np.load('./exp/eval/pointbert@20250521-204815/shapenet_feat.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/pointbert_retrieval_so3@20250521-210850/modelnet40_feat_so3.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print('TAMM PointBERT - No rot/SO3:', mAP)
# TAMM-PointBERT - So3/no rotation
shapenet_data = np.load('./exp/eval/pointbert_retrieval_so3@20250521-210850/shapenet_feat_so3.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/pointbert@20250521-204815/modelnet40_feat.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print('TAMM PointBERT - SO3/No rot:', mAP)

# TAMM-PointBERT - SO3/SO3
shapenet_data = np.load('./exp/eval/pointbert_retrieval_so3@20250521-210850/shapenet_feat_so3.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/pointbert_retrieval_so3@20250521-210850/modelnet40_feat_so3.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
# TAMM-PointBERT - SO3/SO3 (ensemble)
shapenet_data = np.load('./exp/eval/pointbert_retrieval_ensemble_so3@20250522-105707/shapenet_feat_so3.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/pointbert_retrieval_ensemble_so3@20250522-105707/modelnet40_feat_so3.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print('TAMM PointBERT - SO3/SO3 (ensemble):', mAP)

# RI-Mamba - no rotation
shapenet_data = np.load('./exp/eval/ri_mamba_color_retrieval@20250521-211453/shapenet_feat.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/ri_mamba_color_retrieval@20250521-211453/modelnet40_feat.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)

# no rot/so3
shapenet_data = np.load('./exp/eval/ri_mamba_color_retrieval@20250521-211453/shapenet_feat.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/ri_mamba_color_retrieval_so3@20250521-212123/modelnet40_feat_so3.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print('RiMamba - no rot/so3:', mAP)
# so3/no rot
shapenet_data = np.load('./exp/eval/ri_mamba_color_retrieval_so3@20250521-212123/shapenet_feat_so3.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/ri_mamba_color_retrieval@20250521-211453/modelnet40_feat.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print('RiMamba - so3/no rot:', mAP)


# RI-Mamba - SO3/SO3
shapenet_data = np.load('./exp/eval/ri_mamba_color_retrieval_so3@20250521-212123/shapenet_feat_so3.npy', allow_pickle=True).item()
modelnet_data = np.load('./exp/eval/ri_mamba_color_retrieval_so3@20250521-212123/modelnet40_feat_so3.npy', allow_pickle=True).item()

_, class_mAP, mAP = compute_retrieval_mAP(
    shapenet_data['features'], modelnet_data['features'],
    shapenet_data['categories'], modelnet_data['categories']
)
print(mAP)
