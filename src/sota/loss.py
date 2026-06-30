import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = torch.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        if targets.dim() == 1:
            targets = torch.nn.functional.one_hot(targets, num_classes=inputs.size(1)).float()
            
        focal_weight = (1.0 - probs) ** self.gamma
        loss = -targets * focal_weight * log_probs
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            loss = loss * alpha.unsqueeze(0)
            
        loss = torch.sum(loss, dim=1)
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: shape (B, D)
        # labels: shape (B)
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        # Normalize features to unit sphere
        features = F.normalize(features, p=2, dim=1)
        
        # Similarity matrix (B, B)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # For numerical stability, subtract the max similarity value in each row
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Mask out self-contrast (diagonal elements)
        mask = torch.eye(B, dtype=torch.bool, device=device)
        logits_mask = ~mask
        
        # Find positive masks (samples with the same label but not self)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).to(device) & logits_mask
        
        # Check if there are any positive pairs in the batch
        num_pos_per_row = pos_mask.sum(dim=1)
        valid_rows = num_pos_per_row > 0
        
        if not valid_rows.any():
            # If no positive pairs exist in this batch, return 0 loss
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        # Filter logits and masks to valid rows only
        logits = logits[valid_rows]
        pos_mask = pos_mask[valid_rows]
        logits_mask = logits_mask[valid_rows]
        
        # Calculate denominator: sum of exp of all logits (excluding self)
        exp_logits = torch.exp(logits) * logits_mask
        denominator = torch.sum(exp_logits, dim=1, keepdim=True) + 1e-8
        
        # Calculate log probabilities
        log_prob = logits - torch.log(denominator)
        
        # Sum of log probabilities over positive pairs, divided by number of positive pairs
        mean_log_prob_pos = torch.sum(log_prob * pos_mask, dim=1) / (pos_mask.sum(dim=1) + 1e-8)
        
        # Loss is the negative mean of these probabilities
        loss = -torch.mean(mean_log_prob_pos)
        return loss
