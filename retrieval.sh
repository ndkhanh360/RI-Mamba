

# TEXT2PCD + PCD2PCD (Omni3D)  --------------------------------------------------------------------------------------
# RIMAMBA-SHAPENET +++++++++++++++++++++++++++++++++++++++++++
# |--- No rotation 
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py modelnet40.save_feat=True batch_size=4 \
omniobject3d.data=data/OmniObject3D/omniobject3d_captionfeat.npz \
omniobject3d.save_feat=True omniobject3d.save_text_feat=True \
--exp_dir exp/text2pcd --trial_name text2pcd_ri_mamba_cia_coloromni3d \
--config configs/Pre-Training/ri_mamba.yaml \
--resume exp/train/ri_mamba_color_cia_bs64_coloromni3d@20250529-134602/ckpt/best_omniobject3d_overall.pt
# |--- Rotation
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py modelnet40.save_feat=True batch_size=4 \
omniobject3d.data=data/OmniObject3D/omniobject3d_captionfeat.npz omniobject3d.rotate=so3 \
omniobject3d.save_feat=True omniobject3d.save_text_feat=True \
--exp_dir exp/text2pcd --trial_name text2pcd_ri_mamba_cia_so3_coloromni3d \
--config configs/Pre-Training/ri_mamba.yaml \
--resume exp/train/ri_mamba_color_cia_bs64_coloromni3d@20250529-134602/ckpt/best_omniobject3d_overall.pt

# |--- No rotation - text2shape
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py dataset.train_split=meta_data/text2shape/test_updated.json \
--exp_dir exp/text2shape --trial_name saveresult_text2shape_rimamba_shapenet_testset \
--config configs/Pre-Training/ri_mamba.yaml \
--resume exp/text2shape/finetune_small_rimamba_finetune_testtext2shape_bs256@20250714-165215/ckpt/epoch_76.pt  

# RIMAMBA-ENSEMBLE
# |--- No rotation 
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py modelnet40.save_feat=True batch_size=4 \
omniobject3d.data=data/OmniObject3D/omniobject3d_captionfeat.npz \
omniobject3d.save_feat=True omniobject3d.save_text_feat=True \
--exp_dir exp/text2pcd --trial_name text2pcd_ri_mamba_color_cia_bs256_coloromni3d_ensemble_lvis \
--config configs/Pre-Training/ri_mamba.yaml \
--resume exp/train/ri_mamba_color_cia_bs256_coloromni3d_ensemble_lvis@20250601-115317/ckpt/best_omniobject3d_overall.pt
# |--- Rotation
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py modelnet40.save_feat=True batch_size=4 \
omniobject3d.data=data/OmniObject3D/omniobject3d_captionfeat.npz omniobject3d.rotate=so3 \
omniobject3d.save_feat=True omniobject3d.save_text_feat=True \
--exp_dir exp/text2pcd --trial_name text2pcd_ri_mamba_color_cia_bs256_so3_coloromni3d_ensemble_lvis \
--config configs/Pre-Training/ri_mamba.yaml \
--resume exp/train/ri_mamba_color_cia_bs256_coloromni3d_ensemble_lvis@20250601-115317/ckpt/best_omniobject3d_overall.pt

# |--- No rotation - text2shape
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py dataset.train_split=meta_data/text2shape/test.json \
--exp_dir exp/text2shape --trial_name text2shape_rimamba_ensemble_testset \
--config configs/Pre-Training/ri_mamba.yaml \
--resume exp/train/ri_mamba_color_cia_bs256_coloromni3d_ensemble_lvis@20250601-115317/ckpt/best_omniobject3d_overall.pt