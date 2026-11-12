import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
import wandb
from numpy import *
from tqdm import tqdm
import json 

from trainers.trainer_utils import merge_two_branch_results_dist
from utils.locotrans_utils import get_graph_feature # this one is different than the one used in LocoTrans model
from text2pcd import multitext_to_pcd

class TAMM_Trainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, text_alignment_adapter, image_alignment_adapter, clip_adapter,
                 optimizer,
                 scheduler, train_loader, \
                 modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, omniobject3d_loader=None, text2shape_loaders=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.image_alignment_adapter = image_alignment_adapter
        self.text_alignment_adapter = text_alignment_adapter
        self.clip_adapter = clip_adapter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.omniobject3d_loader = omniobject3d_loader
        self.text2shape_loaders = text2shape_loaders
        self.epoch = 0
        self.step = 0
        self.alpha = 0.5
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.best_omniobject3d_acc = 0
        self.best_ndcg = 0
        self.best_ndcg_so3 = 0
        self.config.ngpu = dist.get_world_size()
        self.loss_func = self.smoothl1_contras_loss if config.training.get("contrast_type", "contrast") == "smoothl1" else self.contras_loss
        
    def load_from_checkpoint(self, path, only_model=False):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.image_alignment_adapter.load_state_dict(checkpoint['image_alignment_adapter'])
        self.text_alignment_adapter.load_state_dict(checkpoint['text_alignment_adapter'])

        if not only_model:
            # only_model=True when we want to init the model from a weight and fine-tune
            # opposite to resume training from a checkpoint 
            self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.config.training.scheduler == "default":
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.epoch = checkpoint['epoch'] + 1
            self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            # i=5
            # if i<4:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
            # print("logit", logits.shape, self.rank)
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def smoothl1_contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
        else:
            all_feat1 = feat1
            all_feat2 = feat2

        loss = F.smooth_l1_loss(all_feat1, all_feat2, reduction='sum') / all_feat1.shape[0] # sum across all feat channel 
        
        # For accuracy, you can still normalize temporarily:
        norm_feat1 = F.normalize(all_feat1, dim=1)
        norm_feat2 = F.normalize(all_feat2, dim=1)
        sim_matrix = norm_feat1 @ norm_feat2.T
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        accuracy = (sim_matrix.argmax(dim=1) == labels).float().mean()

        return loss, accuracy
    
    def train_one_epoch(self):
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()
        self.image_alignment_adapter.train()
        self.text_alignment_adapter.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        intra_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader, desc='Training'):
            # print(data['name'])
            # import pdb; pdb.set_trace()
            
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                       device=self.config.device, \
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'], data['features_dense'])
            logit_scale = self.logit_scale(None)

            text_feat = torch.vstack(data['text_feat'])
            img_feat = torch.vstack(data['img_feat'])

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            if self.config.model.name == "IECNN_cls":
                pred_feat, equiv_feat, fuse_feat, [vector, idx] = pred_feat

                # Process pred_feat
                pc_image_feat_pred = self.image_alignment_adapter(pred_feat)
                pc_text_feat_pred = self.text_alignment_adapter(pred_feat)

                # Process equiv_feat
                pc_image_feat_equiv = self.image_alignment_adapter(equiv_feat)
                pc_text_feat_equiv = self.text_alignment_adapter(equiv_feat)

                # Process fuse_feat
                pc_image_feat_fuse = self.image_alignment_adapter(fuse_feat)
                pc_text_feat_fuse = self.text_alignment_adapter(fuse_feat)

                # Now, collect them into lists if you want to loop over them for loss calculation
                all_pc_image_feats = [pc_image_feat_pred, pc_image_feat_equiv, pc_image_feat_fuse]
                all_pc_text_feats = [pc_text_feat_pred, pc_text_feat_equiv, pc_text_feat_fuse]

                text_idx = data['has_text_idx']
                
                for current_pc_image_feat, current_pc_text_feat in zip(all_pc_image_feats, all_pc_text_feats):                                
                    for i in range(self.config.dataset.num_imgs):
                        single_img_feat = img_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                        single_img_feat = single_img_feat.to(self.config.device)

                        # use clip adapter to get new image feature
                        with torch.no_grad():
                            if self.clip_adapter is not None:
                                single_img_feat = self.clip_adapter(single_img_feat)
                                
                        img_contras_loss, img_contras_acc = self.loss_func(current_pc_image_feat, single_img_feat,
                                                                            logit_scale=logit_scale,
                                                                            mask=mask)
                        loss += img_contras_loss * self.config.training.lambda_img_contras
                        img_contras_acc_list.append(img_contras_acc.item())
                    for i in range(self.config.dataset.num_texts):
                        single_text_feat = text_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                        single_text_feat = single_text_feat.to(self.config.device)
                        text_contras_loss, text_contras_acc = self.loss_func(current_pc_text_feat[text_idx], single_text_feat,
                                                                                logit_scale=logit_scale, mask=mask)
                        loss += text_contras_loss * self.config.training.lambda_text_contras

                        text_contras_acc_list.append(text_contras_acc.item())
                                        
                # compute orthogonal loss
                vector = F.normalize(vector, dim=1).permute(0, 3, 2, 1)
                v1 = vector[:, :, 0, :]
                v2 = vector[:, :, 1, :]
                dot_product = (v1 * v2).sum(-1)
                zero_vector = torch.zeros([self.config.batch_size, dot_product.size()[1]], device=dot_product.device)
                loss += self.config.training.loss_orth * F.mse_loss(dot_product, zero_vector)         
                
                # compute consistency loss 
                v1_local = get_graph_feature(v1.transpose(2, 1), idx=idx, k=self.config.model.k)
                v2_local = get_graph_feature(v2.transpose(2, 1), idx=idx, k=self.config.model.k)
                v1_consist = torch.matmul(v1_local, v1_local.transpose(3, 2)).reshape(-1, self.config.model.k)
                v2_consist = torch.matmul(v2_local, v2_local.transpose(3, 2)).reshape(-1, self.config.model.k)
                cl_mask = torch.where(v1_consist.mean(dim=-1) > v2_consist.mean(dim=-1), True, False)
                consistency_loss = F.mse_loss(v1_consist[cl_mask].detach(), v2_consist[cl_mask]) + F.mse_loss(v1_consist[~cl_mask], v2_consist[~cl_mask].detach())
                loss += self.config.training.loss_consist * consistency_loss
            else:            
                pc_image_feat = self.image_alignment_adapter(pred_feat)
                pc_text_feat = self.text_alignment_adapter(pred_feat)

                idx = data['has_text_idx']

                if self.config.training.loss_type == "two_branch":
                    for i in range(self.config.dataset.num_imgs):
                        single_img_feat = img_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                        single_img_feat = single_img_feat.to(self.config.device)

                        # use clip adapter to get new image feature
                        with torch.no_grad():
                            if self.clip_adapter is not None:
                                single_img_feat = self.clip_adapter(single_img_feat)
                        
                        img_contras_loss, img_contras_acc = self.loss_func(pc_image_feat, single_img_feat,
                                                                            logit_scale=logit_scale,
                                                                            mask=mask)
                        # import pdb; pdb.set_trace()
                        loss += img_contras_loss * self.config.training.lambda_img_contras
                        img_contras_acc_list.append(img_contras_acc.item())
                    for i in range(self.config.dataset.num_texts):
                        single_text_feat = text_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                        single_text_feat = single_text_feat.to(self.config.device)
                        text_contras_loss, text_contras_acc = self.loss_func(pc_text_feat[idx], single_text_feat,
                                                                                logit_scale=logit_scale, mask=mask)
                        loss += text_contras_loss * self.config.training.lambda_text_contras

                        text_contras_acc_list.append(text_contras_acc.item())

            loss.backward()
            self.optimizer.step()
            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()
                
            if self.rank == 0 and self.step % (self.config.training.log_freq) == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/text_contras_loss": text_contras_loss.item() if len(idx) > 0 else 0,
                    "train/img_contras_loss": img_contras_loss.item(),
                    # "train/img_text_contras_loss": img_text_contras_loss.item(),
                    # "train/pc_img_to_text_contras_loss": pc_img_to_text_contras_loss.item(),
                    "train/text_contras_acc": 100*text_contras_acc.item() if len(idx) > 0 else 0,
                    "train/img_contras_acc": 100*img_contras_acc.item(),
                    # "train/img_text_contras_acc": img_text_contras_acc.item(),
                    # "train/pc_img_to_text_contras_acc": pc_img_to_text_contras_acc.item(),
                    "train/lr": self.optimizer.param_groups[0]['lr'],
                    "train/epoch": self.epoch,
                    "train/step": self.step,
                    # "train/image_logit_scale": image_logit_scale,
                    # "train/text_logit_scale": text_logit_scale,
                    # "train/img_text_logit_scale": img_text_logit_scale,
                    # "train/has_text": len(idx),
                    # "train/filtered_pair": (mask == False).sum() if mask is not None else 0
                })            

        if self.rank == 0:
            logging.info('Train: text_contras_acc: {0} image_contras_acc: {1}' \
                         .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                                 np.mean(img_contras_acc_list)))

    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "optimizer": self.optimizer.state_dict(),
            "image_alignment_adapter": self.image_alignment_adapter.state_dict(),
            "text_alignment_adapter": self.text_alignment_adapter.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()

            if epoch > self.config.training.test_epoch:
                # self.test_text2shape()
                # self.test_text2shape(rotate='so3')                
                self.test_modelnet40()
                # self.test_objaverse_lvis()
                # self.test_omniobject3d()
                # self.test_scanobjectnn()
                
            # if self.rank == 0:
            # self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))
                
    def test_text2shape(self, rotate=''):
        self.model.eval()
        feat2text_all = []
        ids_all = []

        loader = self.text2shape_loaders[0] if not rotate else self.text2shape_loaders[1]
        
        with torch.no_grad():
            for data in tqdm(loader, desc=f"Text2Shape {rotate}"):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                    
                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)                
                ids_all += data['name'] 
                feat2text_all.append(pred_feat_text.detach().cpu().numpy().astype(np.float32))   
        
        # stack all pcd feats into a single np array
        pcd_feat = np.vstack(feat2text_all)
        # load caption feat      
        clip_path = 'meta_data/text2shape/clip_test_updated.npy'
        txt_clip = np.load(clip_path, allow_pickle=True)
        txt_ids = []
        txt_feat = []
        raw_caps = []
        for obj in txt_clip:
            txt_feat.append(obj['prompt_avg'])
            txt_ids.append(obj['id'])
            raw_caps.append(obj['raw_cap'])
        txt_feat= np.concatenate(txt_feat)
        r1, r5, r10, medr, meanr, ndcg_val, ranks_dis = multitext_to_pcd(txt_feat, pcd_feat, txt_ids, ids_all, captions_list=raw_caps, return_ranks=True)

                        
        if rotate == "":
            if ndcg_val > self.best_ndcg:
                self.best_ndcg = ndcg_val
                self.save_model('best_ndcg')
                logging.info('saving json')
                json.dump(ranks_dis, open('best_results.json', 'w'))
            best_ndcg = self.best_ndcg
        else:
            if ndcg_val > self.best_ndcg_so3:
                self.best_ndcg_so3 = ndcg_val
                self.save_model('best_ndcg_so3')   
                logging.info('saving json')                
                json.dump(ranks_dis, open('best_so3_results.json', 'w'))
                
            best_ndcg = self.best_ndcg_so3         
            
        logging.info(f'Text2Shape {rotate}: R@1: {r1}, R@5: {r5}, NDCG@5: {ndcg_val} ({best_ndcg})')
        
    def save_training_feat(self):
        self.model.eval()
        labels_all = []
        feat_all = []
        ids_all = []
        feat2text_all = [] # this to save the aligned feature for text clip feat 
        
        with torch.no_grad():
            for data in tqdm(self.train_loader, desc="ShapeNet"):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)                
                ids_all += data['name'] 
                labels_all += data['group']
                feat_all.append(pred_feat.detach().cpu())
                feat2text_all.append(pred_feat_text.detach().cpu())
        
        group2category = {
            "02691156": "airplane",
            "02747177": "trash bin",
            "02773838": "bag",
            "02801938": "basket",
            "02808440": "bathtub",
            "02818832": "bed",
            "02828884": "bench",
            "02843684": "birdhouse",
            "02871439": "bookshelf",
            "02876657": "bottle",
            "02880940": "bowl",
            "02924116": "bus",
            "02933112": "cabinet",
            "02942699": "camera",
            "02946921": "can",
            "02954340": "cap",
            "02958343": "car",
            "02992529": "cellphone",
            "03001627": "chair",
            "03046257": "clock",
            "03085013": "keyboard",
            "03207941": "dishwasher",
            "03211117": "display",
            "03261776": "earphone",
            "03325088": "faucet",
            "03337140": "file cabinet",
            "03467517": "guitar",
            "03513137": "helmet",
            "03593526": "jar",
            "03624134": "knife",
            "03636649": "lamp",
            "03642806": "laptop",
            "03691459": "loudspeaker",
            "03710193": "mailbox",
            "03759954": "microphone",
            "03761084": "microwaves",
            "03790512": "motorbike",
            "03797390": "mug",
            "03928116": "piano",
            "03938244": "pillow",
            "03948459": "pistol",
            "03991062": "flowerpot",
            "04004475": "printer",
            "04074963": "remote",
            "04090263": "rifle",
            "04099429": "rocket",
            "04225987": "skateboard",
            "04256520": "sofa",
            "04330267": "stove",
            "04379243": "table",
            "04401088": "telephone",
            "04460130": "tower",
            "04468005": "train",
            "04530566": "watercraft",
            "04554684": "washer"
        }        
        
        save_data = {
            'features': torch.cat(feat_all).numpy().astype(np.float32),
            'features2text': torch.cat(feat2text_all).numpy().astype(np.float32),
            'categories': [group2category[group] for group in labels_all],
            'ids': ids_all
        }
        file_name = 'shapenet_feat' + ('.npy' if not self.train_loader.dataset.rotate else '_so3.npy')
        save_path = os.path.join(self.config.exp_dir, self.config.trial_name, file_name)
        np.save(save_path, save_data)       
                                                    
    def test_modelnet40(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).cuda()
        per_cat_count = torch.zeros(40).cuda()
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        feat_all = []
        with torch.no_grad():
            for data in tqdm(self.modelnet40_loader, desc="ModelNet40"):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                    
                if self.config.model.name == "IECNN_cls":
                    pred_feat = pred_feat[0]

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)
                feat_all.append(pred_feat)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:
            # TODO save to file here
            # convert to numpy, map cls idx to category 
            if self.config.modelnet40.get("save_feat", False):
                save_data = {
                    'features': torch.cat(feat_all).cpu().numpy().astype(np.float32),
                    'categories': [idx2category[idx] for idx in labels_all.numpy()]
                }
                file_name = 'modelnet40_feat' + ('.npy' if not self.modelnet40_loader.dataset.rotate else '_so3.npy')
                save_path = os.path.join(self.config.exp_dir, self.config.trial_name, file_name)
                np.save(save_path, save_data)

            logits_all = logits_text_all + self.alpha * logits_image_all
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            for i in range(40):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            # save per class accuracy
            class_accuracies = {idx2category[i]: per_cat_acc[i].item() for i in range(40) if per_cat_count[i] > 0}
            import csv
            save_path_acc = os.path.join(self.config.exp_dir, self.config.trial_name, 'modelnet40_class_acc.csv')
            with open(save_path_acc, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class', 'acc'])
                for cls_name, acc in class_accuracies.items():
                    writer.writerow([cls_name, acc])
            logging.info(f"Saved per-class accuracy to {save_path_acc}")
            
            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
                self.save_model('best_modelnet40_overall')
            if per_cat_acc.mean() > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = per_cat_acc.mean()
                self.save_model('best_modelnet40_class')

            logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
                                                                                             self.best_modelnet40_overall_acc,
                                                                                             per_cat_acc.mean(),
                                                                                             self.best_modelnet40_class_acc))
            logging.info(
                'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                    topk_acc[1].item(),
                                                                                    topk_acc[2].item()))
                        
            wandb.log({"test_modelnet40/epoch": self.epoch,
                       "test_modelnet40/step": self.step,
                       "test_modelnet40/overall_acc": 100 * overall_acc,
                       "test_modelnet40/class_acc": 100 * per_cat_acc.mean(),})            

    def test_objaverse_lvis(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).cuda()
        per_cat_count = torch.zeros(1156).cuda()
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader, desc="Objaverse LVIS"):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                if self.config.model.name == "IECNN_cls":
                    pred_feat = pred_feat[0]
                    
                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)
                # print("pred_feat_text", pred_feat_text.shape)
                # print("clip_text_feat", clip_text_feat.shape)
                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "objaverse_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:

            logits_all = 1 * logits_text_all + self.alpha * logits_image_all
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in torch.unique(labels_all):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_lvis_acc:
                self.best_lvis_acc = overall_acc
                self.save_model('best_lvis')

            logging.info(
                'Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                                topk_acc[1].item(),
                                                                                                topk_acc[2].item()))
            wandb.log({"test_lvis/epoch": self.epoch,
                       "test_lvis/step": self.step,
                       "test_lvis/overall_acc": 100 * overall_acc,
                       "test_lvis/class_acc": 100 * per_cat_acc.mean(),})

    def test_omniobject3d(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.omniobject3d_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
            
        per_cat_correct = torch.zeros(216).cuda()
        per_cat_count = torch.zeros(216).cuda()
        category2idx = self.omniobject3d_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        save_feat = self.config.omniobject3d.get("save_feat", False)
        save_text_feat = self.config.omniobject3d.get("save_text_feat", False)
        feat_all = []
        text_feat_all, cap_feat_all = [], []
        with torch.no_grad():
            for data in tqdm(self.omniobject3d_loader, desc="Omniobject3d"):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                # if True:
                #     from fvcore.nn import FlopCountAnalysis, flop_count_table
                #     flops = FlopCountAnalysis(self.model, (data['xyz_dense'], data['features_dense']))
                #     logging.info(flop_count_table(flops, max_depth=1))
                #     cnt = flops.total()
                #     logging.info(f"[#FLOPs] total: {cnt}")
                #     import pdb; pdb.set_trace()
                        
                if self.config.model.name == "IECNN_cls":
                    pred_feat = pred_feat[0]
                    
                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)
                feat_all.append(pred_feat)
                
                # save pred text and caption feat for text2pcd retrieval
                if save_text_feat:
                    text_feat_all.append(pred_feat_text)
                    cap_feat_all.append(data["caption_feat"])

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "omniobject3d_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:
            # TODO save to file here
            # convert to numpy, map cls idx to category 
            if save_feat:
                save_data = {
                    'features': torch.cat(feat_all).cpu().numpy().astype(np.float32),
                    'categories': [idx2category[idx] for idx in labels_all.numpy()]
                }
                
                if save_text_feat:
                    save_data['text_feat'] = torch.cat(text_feat_all).cpu().numpy().astype(np.float32)
                    caption_feat_type = self.config.omniobject3d.get("caption_feat", False)
                    if caption_feat_type != False:
                        save_data[caption_feat_type] = torch.cat(cap_feat_all).cpu().numpy().astype(np.float32)
                
                file_name = 'omniobject3d_feat' + self.omniobject3d_loader.dataset.split + ('.npy' if not self.omniobject3d_loader.dataset.rotate else '_so3.npy')
                save_path = os.path.join(self.config.exp_dir, self.config.trial_name, file_name)
                np.save(save_path, save_data)

            logits_all = logits_text_all + self.alpha * logits_image_all
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            for i in range(216):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_omniobject3d_acc:
                self.best_omniobject3d_acc = overall_acc
                self.save_model('best_omniobject3d_overall')


            logging.info('Test omniobject3d: overall acc: {0}({1}) class_acc: {2}'.format(overall_acc,
                                                                                             self.best_omniobject3d_acc,
                                                                                             per_cat_acc.mean()))
            logging.info(
                'Test omniobject3d: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                    topk_acc[1].item(),
                                                                                    topk_acc[2].item()))
                        
            wandb.log({"test_omniobject3d/epoch": self.epoch,
                       "test_omniobject3d/step": self.step,
                       "test_omniobject3d/overall_acc": 100 * overall_acc,
                       "test_omniobject3d/class_acc": 100 * per_cat_acc.mean(),})      
            
    def test_scanobjectnn(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.scanobjectnn_loader, desc="ScanObjectNN"):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                if self.config.model.name == "IECNN_cls":
                    pred_feat = pred_feat[0]
                    
                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:

            logits_all = 1 * logits_text_all + self.alpha * logits_image_all

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in range(15):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            logging.info(
                'Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                               topk_acc[1].item(),
                                                                                               topk_acc[2].item()))
            wandb.log({"test_scanobjectnn/epoch": self.epoch,
                       "test_scanobjectnn/step": self.step,
                       "test_scanobjectnn/overall_acc": 100 * overall_acc,
                       "test_scanobjectnn/class_acc": 100 * per_cat_acc.mean(), })
