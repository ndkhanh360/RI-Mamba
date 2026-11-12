# ri-mamba: zero-shot evaluation
python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py batch_size=32 \
--exp_dir exp/eval --trial_name ri_mamba_color_bs64 \
--config exp/train/ri_mamba_color@20250519-212526/config.yaml \
--resume exp/train/ri_mamba_color@20250519-212526/ckpt/best_lvis.pt

python -m torch.distributed.run --master_port 26789 --nproc_per_node=1 \
test.py batch_size=32 modelnet40.rotate=so3 objaverse_lvis.rotate=so3 \
--exp_dir exp/eval --trial_name ri_mamba_color_bs64_so3 \
--config exp/train/ri_mamba_color@20250519-212526/config.yaml \
--resume exp/train/ri_mamba_color@20250519-212526/ckpt/best_lvis.pt