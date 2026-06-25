import os
import urllib.request
import torch
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
        found = False
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
                    found = True
                else:
                    print(f"Shape mismatch for {target_k}: checkpoint {v.shape} vs model {target_state_dict[target_k].shape}. Skipping.")
                break
                
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded {len(new_state_dict)} tensors.")
    if len(missing_keys) > 0:
        # Ignore keys that are naturally not in pretrained checkpoint (like lstm, fc)
        print(f"Missing keys (not loaded): {len(missing_keys)}")
    return model

class BaselineCNNPANN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=True):
        super(BaselineCNNPANN, self).__init__()
        self.backbone = Cnn14Backbone(in_channels=in_channels)
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_final = nn.Linear(2048, num_classes, bias=True)
        
        self.init_weights()
        
        if pretrained:
            checkpoint_path = download_panns_weights()
            load_panns_state_dict(self, checkpoint_path, in_channels=in_channels)
            
    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc_final)
        
    def forward(self, x):
        # x shape: (B, in_channels, 128, 128)
        x = self.backbone(x) # Shape: (B, 2048, 4, 4)
        
        # Global pooling similar to original Cnn14
        x = torch.mean(x, dim=3) # Shape: (B, 2048, 4)
        (x1, _) = torch.max(x, dim=2) # Shape: (B, 2048)
        x2 = torch.mean(x, dim=2) # Shape: (B, 2048)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        logits = self.fc_final(x)
        return logits

class CNNLSTMPANN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, pretrained=True):
        super(CNNLSTMPANN, self).__init__()
        self.backbone = Cnn14Backbone(in_channels=in_channels)
        
        # BiLSTM input_size is 2048 * 4 = 8192
        self.lstm = nn.LSTM(
            input_size=2048 * 4,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Classifier Head (BiLSTM output size is hidden_size * 2 = 512)
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        if pretrained:
            checkpoint_path = download_panns_weights()
            load_panns_state_dict(self, checkpoint_path, in_channels=in_channels)
            
    def forward(self, x):
        # x shape: (B, in_channels, 128, 128)
        spatial_maps = self.backbone(x) # Shape: (B, 2048, 4, 4)
        b, c, h, w = spatial_maps.shape
        
        # Format sequence: (B, Time_Steps, Feature_Dim)
        # Treat width (axis 3) as the sequence length (4 steps)
        seq_features = spatial_maps.permute(0, 3, 1, 2).contiguous() # (B, 4, 2048, 4)
        seq_features = seq_features.view(b, w, c * h) # (B, 4, 8192)
        
        # LSTM
        lstm_out, _ = self.lstm(seq_features) # (B, 4, 512)
        
        # Final step state
        final_state = lstm_out[:, -1, :] # Shape: (B, 512)
        
        # Classification
        logits = self.fc(final_state)
        return logits
