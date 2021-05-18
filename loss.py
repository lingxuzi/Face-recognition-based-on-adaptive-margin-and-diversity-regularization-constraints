from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class AdaptiveMarginLogits(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=64):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale
        
        self.kernel = Parameter(torch.Tensor(feat_dim, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        
        self.margins = Parameter(torch.Tensor(1, num_classes))
        self.margins.data.fill_(0.5)
    
    def compute_adaptive_margin_loss(self, label):
        mm = self.margins[:, label]
        return torch.sigmoid(-mm.mean())
        # Lm = -1* torch.sum(self.margins[:, label], dim=1)/self.num_classes + 1
        # return Lm
    
    def compute_div_reg_loss(self):
        beta = 10
        
        # k = 2.0 - torch.sigmoid((self.margins - 0.5) * beta)
        k = 2.0 - torch.sigmoid((self.margins - 0.5) * beta)
        
        # kernel_norm = l2_norm(self.kernel, axis=0)
        
        sim = torch.mm(self.kernel.t(), self.kernel)
        non_self_mask = sim.int() != 1
        loss = torch.max(sim * non_self_mask, 1)[0] * k
        return loss.sum() / self.num_classes
        
    def forward(self, embeddings, label):
        kernel_norm = F.normalize(self.kernel, dim=0) #l2_norm(self.kernel, axis=0)

        _embeddings = F.normalize(embeddings, dim=1) #l2_norm(embeddings, axis=1)

        cos_theta = torch.mm(_embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        if label is None:
            return cos_theta
        
        target_logit = cos_theta[torch.arange(
                0, embeddings.size(0)), label].view(-1, 1)
        
        self.margins.data.clamp_(0, 1)
        m = self.margins.data[:, label]
        
        cos_m = torch.cos(m).view(-1, 1)
        sin_m = torch.sin(m).view(-1, 1)
        threshold = torch.cos(math.pi - m).view(-1,1)
        mm = sin_m * m.view(-1, 1)
        
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * cos_m - sin_theta * sin_m - 0.2
        
        final_target_logit = torch.where(
                target_logit > threshold, cos_theta_m, target_logit - mm)
        
        output = cos_theta * 1.0
        output.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        
        # extra_loss
        ada_m_loss = self.compute_adaptive_margin_loss(label)
        div_loss = self.compute_div_reg_loss()
        
        return output * self.scale, 8 * (div_loss + ada_m_loss)
        
        