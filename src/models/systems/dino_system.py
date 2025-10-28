from models.backbone.backbone_2d import ResNet18Backbone
from models.backbone.backbone_3d import R3D18Backbone
from models.heads.dino_head import DINOHead

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.register_buffer("center", torch.zeros(1, out_dim))
    
    def forward(self, student_outputs, teacher_outputs):
        # Concatenate student outputs from local crops: [B * Ncrops, D]
        student_logits = torch.cat(student_outputs)
        student_logits = F.log_softmax(student_logits / self.student_temp, dim=-1)

        # Concatenate teacher outputs from global crops: [B * 2, D]
        teacher_logits = torch.cat(teacher_outputs).detach()
        teacher_logits = F.softmax((teacher_logits - self.center) / self.teacher_temp, dim=-1)

        # Cross-entropy loss
        loss = -torch.sum(teacher_logits * student_logits, dim=-1).mean()

        # Update center by moving average
        new_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(0.9).add_(new_center, alpha=0.1)

        return loss


class DINOSystem(nn.Module):
    def __init__(self, student_backbone, teacher_backbone,
                 student_head, teacher_head, config):
        super().__init__()
        # Choose backbone based on config
        if config['model']['type'] == '2d':
            self.student_backbone = ResNet18Backbone(pretrained=config["model"]["pretrained"])
            self.teacher_backbone = ResNet18Backbone(pretrained=config["model"]["pretrained"])
            embed_dim = 512
        else:
            self.student_backbone = R3D18Backbone(pretrained=config["model"]["pretrained"])
            self.teacher_backbone = R3D18Backbone(pretrained=config["model"]["pretrained"])
            embed_dim = 512

        # Loss + hyperparameters
        self.loss_fn = DINOLoss(out_dim=config["model"]["out_dim"],
                                teacher_temp=config["loss"]["teacher_temp"],
                                student_temp=config["loss"]["student_temp"])
        self.momentum = config["training"]["ema_momentum"]

        # Initialize teacher = student
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in list(self.teacher_backbone.parameters()) + list(self.teacher_head.parameters()):
            p.requires_grad = False

    def forward(self, crops):
        # crops = list of tensors from dataset (2 globals + N locals)
        student_out, teacher_out = [], []

        # --- forward pass (student) ---
        for crop in crops:
            feats = self.student_backbone(crop)
            out = self.student_head(feats)
            student_out.append(out)

        # --- forward pass (teacher) ---
        with torch.no_grad():
            for crop in crops[:2]:  # global only
                feats = self.teacher_backbone(crop)
                out = self.teacher_head(feats)
                teacher_out.append(out)

        # --- loss ---
        loss = self.loss_fn(student_out[2:], teacher_out)
        return loss

    @torch.no_grad()
    def update_teacher(self):
        # Exponential moving average update
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data = pt.data * self.momentum + ps.data * (1.0 - self.momentum)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data = pt.data * self.momentum + ps.data * (1.0 - self.momentum)
