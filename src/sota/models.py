import os
import urllib.request
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x

class Cnn14Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super(Cnn14Backbone, self).__init__()
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
    def forward(self, x):
        # input shape: (B, in_channels, 128, 128)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x

def download_panns_weights(checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    dest_path = os.path.join(checkpoint_dir, "Cnn14_mAP=0.431.pth")
    if os.path.exists(dest_path):
        print(f"Pretrained weights already exist at {dest_path}")
        return dest_path
    
    url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    print(f"Downloading pretrained PANNs Cnn14 weights from {url} to {dest_path}...")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
            
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="PANNs Cnn14 Download") as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
        
    print("Download completed successfully!")
    return dest_path

def load_panns_state_dict(model, checkpoint_path, in_channels=3):
    print(f"Loading pretrained PANNs weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if nested in 'model'
    raw_state_dict = state_dict.get('model', state_dict)
    
    target_state_dict = model.state_dict()
    new_state_dict = {}
    loaded_keys = []
    
    for k, v in raw_state_dict.items():
        for target_k in target_state_dict.keys():
            if target_k == k or target_k.endswith('.' + k):
                # Handle first conv layer weight mapping
                if 'conv_block1.conv1.weight' in k and in_channels != 1:
                    print(f"Adapting {k} from 1 input channel to {in_channels} input channels.")
                    # Shape is (out_channels, 1, kernel_size, kernel_size)
                    v = v.repeat(1, in_channels, 1, 1) / in_channels
                
                if v.shape == target_state_dict[target_k].shape:
                    new_state_dict[target_k] = v
                    loaded_keys.append(target_k)
                break
                
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded {len(new_state_dict)} tensors.")
    return model


# =====================================================================
# MULTI-BRANCH ENCODERS & CROSS-ATTENTION FUSION (PHASE 1)
# =====================================================================

class ResNetBranch(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNetBranch, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        # Modify the first conv layer to accept 1 input channel instead of 3
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Extract features up to layer 4 (excluding avgpool and fc)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, x):
        # Input x shape: (B, 1, H, W)
        return self.features(x)  # Output shape: (B, 512, 4, 4)

class MultiBranchResNet(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(MultiBranchResNet, self).__init__()
        self.branches = nn.ModuleList([ResNetBranch(pretrained=pretrained) for _ in range(in_channels)])
        
    def forward(self, x):
        # Input x shape: (B, in_channels, H, W)
        branch_outputs = []
        for i in range(len(self.branches)):
            branch_input = x[:, i:i+1, :, :]  # Extract single channel (B, 1, H, W)
            branch_outputs.append(self.branches[i](branch_input))
        return branch_outputs  # List of num_branches tensors of shape (B, 512, 4, 4)

class PANNSBranch(nn.Module):
    def __init__(self, pretrained=True):
        super(PANNSBranch, self).__init__()
        self.features = Cnn14Backbone(in_channels=1)
        if pretrained:
            checkpoint_path = download_panns_weights()
            load_panns_state_dict(self.features, checkpoint_path, in_channels=1)
            
    def forward(self, x):
        # Input x shape: (B, 1, H, W)
        return self.features(x)  # Output shape: (B, 2048, 4, 4)

class MultiBranchPANNs(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(MultiBranchPANNs, self).__init__()
        self.branches = nn.ModuleList([PANNSBranch(pretrained=pretrained) for _ in range(in_channels)])
        
    def forward(self, x):
        # Input x shape: (B, in_channels, H, W)
        branch_outputs = []
        for i in range(len(self.branches)):
            branch_input = x[:, i:i+1, :, :]  # Extract single channel (B, 1, H, W)
            branch_outputs.append(self.branches[i](branch_input))
        return branch_outputs  # List of num_branches tensors of shape (B, 2048, 4, 4)

class CrossAttentionFusion(nn.Module):
    def __init__(self, num_branches, feature_dim, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.num_branches = num_branches
        self.feature_dim = feature_dim
        
        # Learned branch embeddings to distinguish Mel, CQT, CWT
        self.branch_embed = nn.Parameter(torch.zeros(1, num_branches, 1, feature_dim))
        # Learned spatial position embeddings (assuming 4x4 spatial resolution = 16 steps)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 16, feature_dim))
        
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * feature_dim, feature_dim)
        )
        
        # Projection layer to fuse branches back to a single feature dimension
        self.project = nn.Linear(num_branches * feature_dim, feature_dim)
        self.norm_final = nn.LayerNorm(feature_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.branch_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, branch_tensors):
        # branch_tensors: list of num_branches tensors of shape (B, C, H, W)
        B, C, H, W = branch_tensors[0].shape
        
        # 1. Flatten and format each branch output to (B, 16, C)
        formatted_branches = []
        for i, x in enumerate(branch_tensors):
            x_flat = x.flatten(2).permute(0, 2, 1)  # (B, 16, C)
            # Add branch and spatial embeddings
            x_flat = x_flat + self.branch_embed[:, i, :, :] + self.pos_embed[:, 0, :, :]
            formatted_branches.append(x_flat)
            
        # Stack to (B, num_branches, 16, C)
        stacked = torch.stack(formatted_branches, dim=1)
        
        # 2. Reshape to combine branches and sequence steps for self-attention
        # Shape: (B, num_branches * 16, C)
        combined = stacked.view(B, self.num_branches * 16, C)
        
        # 3. Multi-Head Attention
        attn_out, _ = self.attn(combined, combined, combined)
        combined = self.norm1(combined + attn_out)
        
        # 4. MLP Block
        mlp_out = self.mlp(combined)
        combined = self.norm2(combined + mlp_out)
        
        # 5. Split back to branches and project
        # Shape back to: (B, num_branches, 16, C)
        split_branches = combined.view(B, self.num_branches, 16, C)
        
        # Permute to (B, 16, num_branches * C) for late fusion projection
        fused = split_branches.permute(0, 2, 1, 3).reshape(B, 16, self.num_branches * C)
        
        # Project back to C
        fused_projected = self.project(fused)
        fused_projected = self.norm_final(fused_projected)  # (B, 16, C)
        
        # 6. Reshape back to spatial representation (B, C, H, W)
        fused_spatial = fused_projected.permute(0, 2, 1).view(B, C, H, W)
        return fused_spatial


# =====================================================================
# NEW: CONFORMER ARCHITECTURE BLOCKS (PHASE 2)
# =====================================================================

class FeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.ff(x)

class ConformerConvModule(nn.Module):
    def __init__(self, d_model, kernel_size=15, expansion_factor=2, dropout=0.1):
        super(ConformerConvModule, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * expansion_factor, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model * expansion_factor, 
            d_model * expansion_factor, 
            kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2,
            groups=d_model * expansion_factor
        )
        self.bn = nn.BatchNorm1d(d_model * expansion_factor)
        self.pointwise_conv2 = nn.Conv1d(d_model * expansion_factor, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, D)
        x_norm = self.norm(x)
        x_conv = x_norm.transpose(1, 2)  # (B, D, T)
        x_conv = F.silu(self.pointwise_conv1(x_conv))
        x_conv = self.depthwise_conv(x_conv)
        x_conv = F.silu(self.bn(x_conv))
        x_conv = self.pointwise_conv2(x_conv)
        x_conv = self.dropout(x_conv)
        return x_conv.transpose(1, 2)  # (B, T, D)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, kernel_size=15, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ffn1 = FeedForward(d_model, expansion_factor=4, dropout=dropout)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size=kernel_size, dropout=dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, expansion_factor=4, dropout=dropout)
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        # Macaron FFN
        x = x + 0.5 * self.ffn1(x)
        # Self-Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm_attn(x + attn_out)
        # Convolution Module
        conv_out = self.conv(x)
        x = self.norm_conv(x + conv_out)
        # FFN
        x = x + 0.5 * self.ffn2(x)
        return self.norm_final(x)


# =====================================================================
# SOTA MODELS DEFINITION
# =====================================================================

class BaselineCNNPANN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=True, multitask=False, cross_attention=False):
        super(BaselineCNNPANN, self).__init__()
        self.multitask = multitask
        self.cross_attention = cross_attention
        
        if cross_attention:
            self.multi_branch = MultiBranchPANNs(in_channels=in_channels, pretrained=pretrained)
            self.fusion = CrossAttentionFusion(num_branches=in_channels, feature_dim=2048)
        else:
            self.backbone = Cnn14Backbone(in_channels=in_channels)
            if pretrained:
                checkpoint_path = download_panns_weights()
                load_panns_state_dict(self.backbone, checkpoint_path, in_channels=in_channels)
                
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_final = nn.Linear(2048, num_classes, bias=True)
        
        if multitask:
            self.fc_pathology = nn.Linear(2048, 3, bias=True)
            
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc_final)
        if self.multitask:
            init_layer(self.fc_pathology)
            
    def forward(self, x):
        if self.cross_attention:
            spatial_maps = self.fusion(self.multi_branch(x))
        else:
            spatial_maps = self.backbone(x)  # Shape: (B, 2048, 4, 4)
            
        # Global pooling similar to original Cnn14
        x = torch.mean(spatial_maps, dim=3)  # Shape: (B, 2048, 4)
        (x1, _) = torch.max(x, dim=2)  # Shape: (B, 2048)
        x2 = torch.mean(x, dim=2)  # Shape: (B, 2048)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x_shared = F.relu_(self.fc1(x))
        x_shared = F.dropout(x_shared, p=0.5, training=self.training)
        
        logits_cycle = self.fc_final(x_shared)
        if self.multitask:
            logits_pathology = self.fc_pathology(x_shared)
            return logits_cycle, logits_pathology
            
        return logits_cycle

class CNNLSTMPANN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=True, multitask=False, cross_attention=False, sequence_len=4, use_conformer=False):
        super(CNNLSTMPANN, self).__init__()
        self.multitask = multitask
        self.cross_attention = cross_attention
        self.sequence_len = sequence_len
        self.use_conformer = use_conformer
        
        if cross_attention:
            self.multi_branch = MultiBranchPANNs(in_channels=in_channels, pretrained=pretrained)
            self.fusion = CrossAttentionFusion(num_branches=in_channels, feature_dim=2048)
        else:
            self.backbone = Cnn14Backbone(in_channels=in_channels)
            if pretrained:
                checkpoint_path = download_panns_weights()
                load_panns_state_dict(self.backbone, checkpoint_path, in_channels=in_channels)
                
        # High-res sequence projection
        if sequence_len > 4:
            self.time_project = nn.Linear(16, sequence_len)
            seq_dim = 2048
        else:
            seq_dim = 2048 * 4
            
        if use_conformer:
            self.feat_project = nn.Linear(seq_dim, 256)
            self.conformer = nn.Sequential(*[ConformerBlock(d_model=256, num_heads=4) for _ in range(2)])
            fc_input_dim = 256
        else:
            self.lstm = nn.LSTM(
                input_size=seq_dim,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.3
            )
            fc_input_dim = 512
            
        # Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        if multitask:
            self.fc_pathology = nn.Sequential(
                nn.Linear(fc_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3)
            )
            
    def forward(self, x):
        if self.cross_attention:
            spatial_maps = self.fusion(self.multi_branch(x))
        else:
            spatial_maps = self.backbone(x)  # Shape: (B, 2048, 4, 4)
            
        b, c, h, w = spatial_maps.shape
        
        if self.sequence_len > 4:
            # Flatten spatial dimensions
            seq_feats = spatial_maps.flatten(2)  # (B, C, 16)
            # Project time dimension to sequence_len
            seq_feats = self.time_project(seq_feats)  # (B, C, sequence_len)
            # Transpose to (B, sequence_len, C)
            seq_features = seq_feats.transpose(1, 2)
        else:
            # Baseline mode (sequence length 4, features 8192)
            seq_features = spatial_maps.permute(0, 3, 1, 2).contiguous()
            seq_features = seq_features.view(b, w, c * h)
            
        if self.use_conformer:
            seq_features = self.feat_project(seq_features)
            conformer_out = self.conformer(seq_features)  # (B, sequence_len, 256)
            final_state = conformer_out.mean(dim=1)  # Mean pooling over time sequence (B, 256)
        else:
            lstm_out, _ = self.lstm(seq_features)
            final_state = lstm_out[:, -1, :]  # Last step (B, 512)
            
        logits_cycle = self.fc(final_state)
        
        if self.multitask:
            logits_pathology = self.fc_pathology(final_state)
            return logits_cycle, logits_pathology
            
        return logits_cycle

class BaselineCNNResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=True, multitask=False, cross_attention=False):
        super(BaselineCNNResNet, self).__init__()
        self.multitask = multitask
        self.cross_attention = cross_attention
        
        if cross_attention:
            self.multi_branch = MultiBranchResNet(in_channels=in_channels, pretrained=pretrained)
            self.fusion = CrossAttentionFusion(num_branches=in_channels, feature_dim=512)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            if multitask:
                self.fc_pathology = nn.Linear(512, 3)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.resnet = models.resnet18(weights=weights)
            if in_channels != 3:
                self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, num_classes)
            if multitask:
                self.fc_pathology = nn.Linear(num_ftrs, 3)
            
    def forward(self, x):
        if self.cross_attention:
            branch_feats = self.multi_branch(x)  # List of (B, 512, 4, 4)
            fused_feats = self.fusion(branch_feats)  # (B, 512, 4, 4)
            pooled = self.avgpool(fused_feats)  # (B, 512, 1, 1)
            x_flat = torch.flatten(pooled, 1)  # (B, 512)
            logits_cycle = self.fc(x_flat)
            if self.multitask:
                logits_pathology = self.fc_pathology(x_flat)
                return logits_cycle, logits_pathology
            return logits_cycle
        else:
            if not self.multitask:
                return self.resnet(x)
                
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)

            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)

            x = self.resnet.avgpool(x)
            x = torch.flatten(x, 1)
            
            logits_cycle = self.resnet.fc(x)
            logits_pathology = self.fc_pathology(x)
            return logits_cycle, logits_pathology

class CNNLSTMResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=True, multitask=False, cross_attention=False, sequence_len=4, use_conformer=False):
        super(CNNLSTMResNet, self).__init__()
        self.multitask = multitask
        self.cross_attention = cross_attention
        self.sequence_len = sequence_len
        self.use_conformer = use_conformer
        
        if cross_attention:
            self.multi_branch = MultiBranchResNet(in_channels=in_channels, pretrained=pretrained)
            self.fusion = CrossAttentionFusion(num_branches=in_channels, feature_dim=512)
        else:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            resnet = models.resnet18(weights=weights)
            if in_channels != 3:
                resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.spatial_extractor = nn.Sequential(*list(resnet.children())[:-2])
            
        # High-res sequence projection
        if sequence_len > 4:
            self.time_project = nn.Linear(16, sequence_len)
            seq_dim = 512
        else:
            seq_dim = 512 * 4
            
        if use_conformer:
            self.feat_project = nn.Linear(seq_dim, 256)
            self.conformer = nn.Sequential(*[ConformerBlock(d_model=256, num_heads=4) for _ in range(2)])
            fc_input_dim = 256
        else:
            self.lstm = nn.LSTM(
                input_size=seq_dim, 
                hidden_size=256, 
                num_layers=2, 
                bidirectional=True, 
                batch_first=True, 
                dropout=0.3
            )
            fc_input_dim = 512
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        if multitask:
            self.fc_pathology = nn.Sequential(
                nn.Linear(fc_input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3)
            )
            
    def forward(self, x):
        if self.cross_attention:
            spatial_maps = self.fusion(self.multi_branch(x))
        else:
            spatial_maps = self.spatial_extractor(x)
            
        b, c, h, w = spatial_maps.shape
        
        if self.sequence_len > 4:
            # Flatten spatial dimensions
            seq_feats = spatial_maps.flatten(2)  # (B, C, 16)
            # Project time dimension to sequence_len
            seq_feats = self.time_project(seq_feats)  # (B, C, sequence_len)
            # Transpose to (B, sequence_len, C)
            seq_features = seq_feats.transpose(1, 2)
        else:
            # Baseline mode (sequence length 4, features 2048)
            seq_features = spatial_maps.permute(0, 3, 1, 2).contiguous()
            seq_features = seq_features.view(b, w, c * h)
            
        if self.use_conformer:
            seq_features = self.feat_project(seq_features)
            conformer_out = self.conformer(seq_features)  # (B, sequence_len, 256)
            final_state = conformer_out.mean(dim=1)  # Mean pooling over time sequence (B, 256)
        else:
            lstm_out, _ = self.lstm(seq_features)
            final_state = lstm_out[:, -1, :]
            
        logits_cycle = self.fc(final_state)
        if self.multitask:
            logits_pathology = self.fc_pathology(final_state)
            return logits_cycle, logits_pathology
            
        return logits_cycle
