# train RI-Mamba
python -m torch.distributed.run --master_port 26987 --nproc_per_node=1 \
main.py training.test_epoch=-1 \
--exp_dir exp/train --trial_name ri_mamba_color_cia_bs64_coloromni3d \
--config configs/Pre-Training/ri_mamba.yaml 

# finetune on Text2Shape
python -m torch.distributed.run --master_port 26999 --nproc_per_node=1 \
main.py batch_size=64 training.test_epoch=-1 \
--exp_dir exp/text2shape --trial_name finetune_small_rimamba_finetune_testtext2shape \
--finetune --resume exp/train/ri_mamba_color_cia_bs64_coloromni3d@20250529-134602/ckpt/best_overall.pt \
--config configs/Text2Shape/ri_mamba.yaml 
