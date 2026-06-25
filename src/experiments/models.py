import torch
import torch.nn as nn
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(BaselineCNN, self).__init__()
        # Load backbone with default ImageNet weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer if input channels is not 3
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
        # Modify classification head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

class CNNLSTM(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(CNNLSTM, self).__init__()
        # Load backbone with default ImageNet weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Modify the first conv layer if input channels is not 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
        # Remove average pooling and final fully connected layer
        self.spatial_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Sequential Modeling: Bidirectional LSTM
        # Treat width (time axis = 4 steps) as the sequence dimension
        # Treat channel (512) * height (4) = 2048 as the feature dimension
        self.lstm = nn.LSTM(
            input_size=512 * 4, 
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
        
    def forward(self, x):
        # x shape: (B, C, 128, 128)
        
        # Extract spatial feature maps: output shape is (B, 512, 4, 4)
        spatial_maps = self.spatial_extractor(x)
        b, c, h, w = spatial_maps.shape
        
        # Re-arrange dimensions to format sequence: (B, Time_Steps, Feature_Dim)
        # Treat width (axis 3) as the sequence length (4 steps)
        # Permute shape to: (B, w, c, h) -> (B, 4, 512, 4)
        seq_features = spatial_maps.permute(0, 3, 1, 2).contiguous()
        # Flatten features at each step: (B, 4, 512 * 4) -> (B, 4, 2048)
        seq_features = seq_features.view(b, w, c * h)
        
        # Pass sequence to Bidirectional LSTM
        lstm_out, (h_n, c_n) = self.lstm(seq_features) # Output shape: (B, 4, 512)
        
        # Take output of the final time step
        final_state = lstm_out[:, -1, :] # Shape: (B, 512)
        
        # Classification
        logits = self.fc(final_state)
        return logits
