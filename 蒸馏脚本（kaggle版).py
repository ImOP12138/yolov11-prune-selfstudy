"""
YOLOv11知识蒸馏脚本
用于提升剪枝后模型的性能
Teacher: 原始未剪枝的模型 (baseline)
Student: 剪枝后的模型 (pruned)

蒸馏策略:
1. Logit蒸馏: 分类分支的软标签蒸馏 (KL散度)
2. Feature蒸馏: 中间特征图的蒸馏 (MSE/L2损失)
"""
import os
import sys
import math
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import distributed as dist
from torch import optim

from ultralytics import YOLO
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, attempt_load_weights
from ultralytics.nn.tasks_pruned import DetectionModelPruned
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    colorstr,
    yaml_save,
)
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    de_parallel,
)
from ultralytics.utils.checks import check_imgsz, check_amp
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors
from ultralytics.utils.ops import crop_mask, xywh2xyxy
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results

warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class DistillationLoss:
    """
    知识蒸馏损失函数
    包含:
    1. 硬标签损失 (学生模型的标准检测损失)
    2. Logit蒸馏损失 (分类分支的KL散度)
    3. Feature蒸馏损失 (中间特征图的MSE损失)
    """
    
    def __init__(self, model_student, model_teacher, args, distill_type='logit', 
                 temp=4.0, alpha=0.5, beta=0.0):
        """
        Args:
            model_student: 学生模型 (剪枝后的模型)
            model_teacher: 教师模型 (原始未剪枝模型)
            args: 训练参数
            distill_type: 蒸馏类型, 'logit' / 'feature' / 'both'
            temp: 温度参数, 用于软化logits
            alpha: logit蒸馏损失的权重
            beta: feature蒸馏损失的权重
        """
        self.device = next(model_student.parameters()).device
        self.hyp = args
        
        m_student = model_student.model[-1]
        m_teacher = model_teacher.model[-1]
        
        self.stride = m_student.stride
        self.nc = m_student.nc
        self.reg_max = m_student.reg_max
        self.no = m_student.nc + m_student.reg_max * 4
        
        self.temp = temp
        self.alpha = alpha
        self.beta = beta
        self.distill_type = distill_type
        
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)
        
        self.teacher_model = model_teacher
        self.teacher_model.eval()
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def bbox_decode(self, anchor_points, pred_dist):
        """解码预测的边界框"""
        b, a, c = pred_dist.shape
        proj = self.proj.to(pred_dist.device)
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
            proj.type(pred_dist.dtype)
        )
        return self._dist2bbox(pred_dist, anchor_points, xywh=False)
    
    @staticmethod
    def _dist2bbox(distance, anchor_points, xywh=True):
        """将距离转换为边界框坐标"""
        lt, rb = distance.chunk(2, -1)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), -1)
        return torch.cat((x1y1, x2y2), -1)
    
    def preprocess(self, targets, batch_size, scale_tensor):
        """预处理目标标签"""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def kl_divergence_loss(self, student_logits, teacher_logits, temp):
        """
        计算KL散度损失用于logit蒸馏
        Args:
            student_logits: 学生模型的分类logits
            teacher_logits: 教师模型的分类logits
            temp: 温度参数
        """
        student_soft = F.log_softmax(student_logits / temp, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temp, dim=-1)
        
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temp ** 2)
        return kl_loss
    
    def feature_distillation_loss(self, student_feats, teacher_feats):
        """
        特征蒸馏损失 (MSE)
        Args:
            student_feats: 学生模型的中间特征图列表
            teacher_feats: 教师模型的中间特征图列表
        """
        loss = 0.0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            if s_feat.shape != t_feat.shape:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], mode='bilinear', align_corners=False)
                if s_feat.shape[1] != t_feat.shape[1]:
                    continue
            
            loss += F.mse_loss(s_feat, t_feat)
        
        return loss
    
    def __call__(self, preds_student, preds_teacher, batch, student_feats=None, teacher_feats=None):
        """
        计算总损失
        Args:
            preds_student: 学生模型的预测输出
            preds_teacher: 教师模型的预测输出
            batch: 批次数据
            student_feats: 学生模型的中间特征图 (可选)
            teacher_feats: 教师模型的中间特征图 (可选)
        """
        if isinstance(preds_student, (tuple, list)):
            device = preds_student[0].device
        else:
            device = preds_student.device
        loss = torch.zeros(4, device=device)
        
        feats_student = preds_student[1] if isinstance(preds_student, (tuple, list)) else preds_student
        feats_teacher = preds_teacher[1] if isinstance(preds_teacher, (tuple, list)) else preds_teacher
        
        pred_distri_s, pred_scores_s = torch.cat(
            [xi.view(feats_student[0].shape[0], self.no, -1) for xi in feats_student], 2
        ).split((self.reg_max * 4, self.nc), 1)
        
        pred_distri_t, pred_scores_t = torch.cat(
            [xi.view(feats_teacher[0].shape[0], self.no, -1) for xi in feats_teacher], 2
        ).split((self.reg_max * 4, self.nc), 1)
        
        pred_scores_s = pred_scores_s.permute(0, 2, 1).contiguous()
        pred_distri_s = pred_distri_s.permute(0, 2, 1).contiguous()
        pred_scores_t = pred_scores_t.permute(0, 2, 1).contiguous()
        pred_distri_t = pred_distri_t.permute(0, 2, 1).contiguous()
        
        dtype = pred_scores_s.dtype
        batch_size = pred_scores_s.shape[0]
        device = pred_scores_s.device
        imgsz = torch.tensor(feats_student[0].shape[2:], device=device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats_student, self.stride, 0.5)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)
        
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        pred_bboxes_s = self.bbox_decode(anchor_points, pred_distri_s)
        
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores_s.detach().sigmoid(),
            (pred_bboxes_s.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        loss[1] = self.bce(pred_scores_s, target_scores.to(dtype)).sum() / target_scores_sum
        
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            
            from ultralytics.utils.metrics import bbox_iou
            from ultralytics.utils.tal import bbox2dist
            
            iou = bbox_iou(pred_bboxes_s[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            loss[0] = ((1.0 - iou) * weight).sum() / target_scores_sum
            
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            dfl_loss = self._dfl_loss(pred_distri_s[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            loss[2] = dfl_loss.sum() / target_scores_sum
        
        if self.distill_type in ['logit', 'both']:
            loss_distill_logit = self.kl_divergence_loss(
                pred_scores_s.view(-1, self.nc),
                pred_scores_t.view(-1, self.nc).detach(),
                self.temp
            )
            loss[3] += loss_distill_logit * self.alpha
        
        if self.distill_type in ['feature', 'both'] and student_feats is not None and teacher_feats is not None:
            loss_distill_feat = self.feature_distillation_loss(student_feats, teacher_feats)
            loss[3] += loss_distill_feat * self.beta
        
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        return loss.sum() * batch_size, loss.detach()
    
    def _dfl_loss(self, pred_dist, target):
        """Distribution Focal Loss"""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class DistillationTrainer(BaseTrainer):
    """
    知识蒸馏训练器
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        distill_args = {}
        if overrides:
            distill_args = {
                'teacher_weights': overrides.pop('teacher_weights', None),
                'maskbndict_path': overrides.pop('maskbndict_path', None),
                'distill_type': overrides.pop('distill_type', 'logit'),
                'distill_temp': overrides.pop('distill_temp', 4.0),
                'distill_alpha': overrides.pop('distill_alpha', 0.5),
                'distill_beta': overrides.pop('distill_beta', 0.0),
            }
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_weights = distill_args.get('teacher_weights')
        self.maskbndict_path = distill_args.get('maskbndict_path')
        self.distill_type = distill_args.get('distill_type', 'logit')
        self.distill_temp = distill_args.get('distill_temp', 4.0)
        self.distill_alpha = distill_args.get('distill_alpha', 0.5)
        self.distill_beta = distill_args.get('distill_beta', 0.0)
        self.finetune = True
        self.sr = None
    
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)
    
    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch
    
    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args
    
    def get_model(self, cfg=None, weights=None, verbose=True, maskbndict=None):
        assert maskbndict is not None, "maskbndict must be provided for pruned model"
        model = DetectionModelPruned(maskbndict, cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def setup_model(self):
        if isinstance(self.model, torch.nn.Module):
            return
        
        cfg, weights = self.model, None
        ckpt = None
        
        if str(self.model).endswith(".pt"):
            model_loaded, ckpt = attempt_load_one_weight(self.model)
            cfg = model_loaded.yaml
            maskbndict = ckpt.get('maskbndict', None) if ckpt else None
            weights = model_loaded
            
            if maskbndict is None and self.maskbndict_path:
                LOGGER.info(colorstr('blue', f'Loading maskbndict from {self.maskbndict_path}...'))
                maskbndict_ckpt = torch.load(self.maskbndict_path, map_location='cpu')
                maskbndict = maskbndict_ckpt.get('maskbndict', None) if isinstance(maskbndict_ckpt, dict) else None
        else:
            maskbndict = None
        
        if maskbndict is None:
            raise ValueError("maskbndict is required for pruned model. Please provide --maskbndict-path or use a checkpoint that contains maskbndict.")
        
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1, maskbndict=maskbndict)
        
        LOGGER.info(colorstr('blue', f'Loading teacher model from {self.teacher_weights}...'))
        self.teacher_model = YOLO(self.teacher_weights)
        self.teacher_model.model = self.teacher_model.model.to(self.device)
        self.teacher_model.model.eval()
        
        for param in self.teacher_model.model.parameters():
            param.requires_grad = False
        
        self.distill_loss = DistillationLoss(
            model_student=self.model,
            model_teacher=self.teacher_model.model,
            args=self.args,
            distill_type=self.distill_type,
            temp=self.distill_temp,
            alpha=self.distill_alpha,
            beta=self.distill_beta
        )
        
        return ckpt
    
    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "distill_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=deepcopy(self.args), _callbacks=self.callbacks
        )
    
    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys
    
    def progress_string(self):
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )
    
    def _setup_train(self, world_size):
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        always_freeze_names = [".dfl"]
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        for k, v in self.model.named_parameters():
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{k}'")
                v.requires_grad = False
            elif not v.requires_grad and v.dtype.is_floating_point:
                v.requires_grad = True
        
        self.amp = torch.tensor(self.args.amp).to(self.device)
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup
        if RANK > -1 and world_size > 1:
            dist.broadcast(self.amp, src=0)
        self.amp = bool(self.amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)
        
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs
        
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()
        
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks("on_pretrain_routine_end")
    
    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting distillation training for {self.epochs} epochs...\n'
            f'Teacher model: {self.teacher_weights}\n'
            f'Distillation type: {self.distill_type}\n'
            f'Temperature: {self.distill_temp}, Alpha: {self.distill_alpha}, Beta: {self.distill_beta}'
        )
        
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.start_epoch
        self.optimizer.zero_grad()
        
        while True:
            self.epoch = epoch
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()
            
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                with autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    preds_student = self.model.predict(batch["img"])
                    
                    with torch.no_grad():
                        preds_teacher = self.teacher_model.model.predict(batch["img"])
                    
                    self.loss, self.loss_items = self.distill_loss(preds_student, preds_teacher, batch)
                    
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                
                self.scaler.scale(self.loss).backward()
                
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),
                            batch["cls"].shape[0],
                            batch["img"].shape[-1],
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                
                self.run_callbacks("on_train_batch_end")
            
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks("on_train_epoch_end")
            
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
                
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks("on_model_save")
            
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch
                self.stop |= epoch >= self.epochs
            self.run_callbacks("on_fit_epoch_end")
            self._clear_memory()
            
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
            epoch += 1
        
        if RANK in {-1, 0}:
            LOGGER.info(
                f"\n{epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        self.run_callbacks("teardown")
    
    def save_model(self):
        import io
        
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()
        
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
    
    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
    
    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)
    
    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)


def parse_opt():
    import argparse
    parser = argparse.ArgumentParser(description='YOLOv11 Knowledge Distillation Training')
    
    parser.add_argument('--student-weights', type=str, 
                        default=str(ROOT / '0.13微调后的模型/OT-pruned0.13.pt'),
                        help='剪枝后的学生模型权重路径')
    parser.add_argument('--maskbndict-path', type=str,
                        default=str(ROOT / 'weights-pruned-yolo11n-ot-lastepoch-0.13/pruned_ot.pt'),
                        help='maskbndict文件路径 (如果学生模型权重中不包含maskbndict)')
    parser.add_argument('--teacher-weights', type=str,
                        default=str(ROOT / 'yolo11n-bestbase.pt'),
                        help='教师模型权重路径 (原始未剪枝模型)')
    parser.add_argument('--data', type=str,
                        default='/kaggle/working/visdrone.yaml',
                        help='数据集配置文件路径')
    parser.add_argument('--cfg', type=str,
                        default=str(ROOT / 'ultralytics/cfg/default.yaml'),
                        help='训练配置文件路径')
    parser.add_argument('--project', type=str, default='.',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default='runs/distill',
                        help='实验名称')
    parser.add_argument('--epochs', type=int, default=2,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=12,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=1024,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='0',
                        help='训练设备 (cuda device, i.e. 0 or 0,1,2,3 or cpu)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        help='优化器 (SGD, Adam, AdamW)')
    parser.add_argument('--lr0', type=float, default=1e-4,
                        help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率 (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD动量/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='权重衰减')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                        help='预热轮数')
    parser.add_argument('--warmup-momentum', type=float, default=0.8,
                        help='预热动量')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1,
                        help='预热偏置学习率')
    parser.add_argument('--box', type=float, default=7.5,
                        help='边界框损失增益')
    parser.add_argument('--cls', type=float, default=0.5,
                        help='分类损失增益')
    parser.add_argument('--dfl', type=float, default=1.5,
                        help='DFL损失增益')
    
    parser.add_argument('--distill-type', type=str, default='logit',
                        choices=['logit', 'feature', 'both'],
                        help='蒸馏类型: logit(分类logits蒸馏), feature(特征蒸馏), both(两者结合)')
    parser.add_argument('--distill-temp', type=float, default=4.0,
                        help='蒸馏温度参数 (软化logits)')
    parser.add_argument('--distill-alpha', type=float, default=0.5,
                        help='logit蒸馏损失权重 (0-1)')
    parser.add_argument('--distill-beta', type=float, default=0.0,
                        help='feature蒸馏损失权重 (仅当distill-type为feature或both时有效)')
    
    parser.add_argument('--multi-scale', action='store_true',
                        help='多尺度训练')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='标签平滑')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='自动混合精度训练')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--save-period', type=int, default=1,
                        help='每x轮保存一次模型 (禁用: -1)')
    
    opt = parser.parse_args()
    return opt


def main(opt):
    overrides = {
        'model': opt.student_weights,
        'data': opt.data,
        'cfg': opt.cfg,
        'project': opt.project,
        'name': opt.name,
        'epochs': opt.epochs,
        'batch': opt.batch,
        'imgsz': opt.imgsz,
        'device': opt.device,
        'workers': opt.workers,
        'optimizer': opt.optimizer,
        'lr0': opt.lr0,
        'lrf': opt.lrf,
        'momentum': opt.momentum,
        'weight_decay': opt.weight_decay,
        'warmup_epochs': opt.warmup_epochs,
        'warmup_momentum': opt.warmup_momentum,
        'warmup_bias_lr': opt.warmup_bias_lr,
        'box': opt.box,
        'cls': opt.cls,
        'dfl': opt.dfl,
        'multi_scale': opt.multi_scale,
        'label_smoothing': opt.label_smoothing,
        'resume': opt.resume if opt.resume else False,
        'amp': opt.amp,
        'patience': opt.patience,
        'save_period': opt.save_period,
        'teacher_weights': opt.teacher_weights,
        'maskbndict_path': opt.maskbndict_path,
        'distill_type': opt.distill_type,
        'distill_temp': opt.distill_temp,
        'distill_alpha': opt.distill_alpha,
        'distill_beta': opt.distill_beta,
        'finetune': True,
    }
    
    trainer = DistillationTrainer(overrides=overrides)
    trainer.train()
    
    LOGGER.info(colorstr('green', 'Knowledge distillation training completed!'))
    LOGGER.info(colorstr('green', f'Best model saved to: {trainer.save_dir / "weights" / "best.pt"}'))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
