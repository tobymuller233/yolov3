import torch
import torch.nn as nn

class YOLOFaceNet(nn.Module):
    def __init__(self):
        super(YOLOFaceNet, self).__init__()

        # Define the layers based on the configuration you provided
        self.layer1 = self._conv_block(3, 8, 3, 2) # [convolutional] 3->8 filters, stride 2
        self.layer2 = self._conv_block(8, 8, 3, 1, groups=8) # groups=8
        self.layer3 = self._conv_block(8, 4, 1, 1, activation='linear')
        
        self.layer4 = self._conv_block(4, 24, 1, 1)
        self.layer5 = self._conv_block(24, 24, 3, 2, groups=24)
        self.layer6 = self._conv_block(24, 6, 1, 1, activation='linear')
        
        self.layer7 = self._conv_block(6, 36, 1, 1)
        self.layer8 = self._conv_block(36, 36, 3, 1, groups=36)
        self.layer9 = self._conv_block(36, 6, 1, 1, activation='linear')

        self.shortcut1 = nn.Identity() # shortcut from -4
        
        self.layer10 = self._conv_block(6, 36, 1, 1)
        self.layer11 = self._conv_block(36, 36, 3, 2, groups=36)
        self.layer12 = self._conv_block(36, 8, 1, 1, activation='linear')

        # Add the rest of the layers similarly...
        # Make sure to include the [shortcut] layers and route layers where appropriate

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, groups=1, activation='relu'):
        """Helper function to create a convolutional block."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups),
            nn.BatchNorm2d(out_channels)
        ]
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'linear':
            pass  # No activation for linear
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass with shape printing to check tensor dimensions
        print(f"Input: {x.shape}")
        
        x = self.layer1(x)
        print(f"After layer 1: {x.shape}")
        
        x = self.layer2(x)
        print(f"After layer 2: {x.shape}")
        
        x = self.layer3(x)
        print(f"After layer 3: {x.shape}")
        
        x = self.layer4(x)
        print(f"After layer 4: {x.shape}")
        
        x = self.layer5(x)
        print(f"After layer 5: {x.shape}")
        
        x = self.layer6(x)
        print(f"After layer 6: {x.shape}")
        
        # Apply shortcut for layer9
        shortcut1 = self.shortcut1(x)
        
        x = self.layer7(shortcut1)
        print(f"After layer 7: {x.shape}")
        
        # Continue forward pass for other layers similarly...
        
        return x


# Example usage:
model = YOLOFaceNet()
input_tensor = torch.randn(1, 3, 256, 320)  # Batch size 1, 3 channels, 256x320 image
output = model(input_tensor)

