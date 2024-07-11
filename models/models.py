import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchmetrics
from monai.inferers import SimpleInferer


class classificationModel(pl.LightningModule):
    def __init__(self,  
                 number_classes,
                 model=None,
                 return_activated_output=True):
        
        super(classificationModel, self).__init__()
        
        self.model = model
        self.n_outputs = number_classes
        self.return_activated_output = return_activated_output
  
        if self.n_outputs == 1:
            self.activation = nn.Sigmoid()
        elif self.n_outputs > 1:
            self.activation = nn.Softmax()

    def forward(self, x):

        xOut, x_amp = self.model(x)

        if self.return_activated_output:
            xOut = self.activation(xOut)

        return xOut, x_amp


########################################################################################################################
""" LVO-affected brain hemisphere detection. SAGITTAL projection """
########################################################################################################################
""" UTILS """
class conv_block_3d_anisotropic(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, padding=0, kernel_size=(3,1,1)):
        super(conv_block_3d_anisotropic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation_rate),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation_rate),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)    
        
        return x
   
class encoder3D_anisotropicKernels(pl.LightningModule):
    def __init__(self, input_channels, output_channels, kernel_size=(3,1,1)):
        super(encoder3D_anisotropicKernels, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=kernel_size)

        self.Conv1 = conv_block_3d_anisotropic(in_channels=input_channels, out_channels=output_channels[0], kernel_size=kernel_size)
        self.Conv2 = conv_block_3d_anisotropic(in_channels=output_channels[0], out_channels=output_channels[1], kernel_size=kernel_size)
        self.Conv3 = conv_block_3d_anisotropic(in_channels=output_channels[1], out_channels=output_channels[2], kernel_size=kernel_size)
        
    def forward(self, x):
        
        """ Encoding path """
        x1 = self.Conv1(x) 
        x2 = self.Maxpool(x1) 

        x2 = self.Conv2(x2) 
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)

        return x3 

""" 3D to 2D """
# L3P + 2D CNN (single CB2D)
class L3P_2D_CNN_sagittal(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 num_channels=[8,16,32],
                 n_classes=1):

        super(L3P_2D_CNN_sagittal, self).__init__()

        self.encoder3D_anisotropicKernels_sagittal = encoder3D_anisotropicKernels(input_channels=input_channels, output_channels=[num_channels[0], num_channels[1], num_channels[2]], kernel_size=(3,1,1))

        self.anisotropicMaxpool = nn.MaxPool3d(kernel_size=(10,1,1),stride=(10,1,1))

        self.conv = nn.Conv2d(num_channels[2], num_channels[2], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(num_channels[2], n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1)

        # 3D encoder
        x = self.encoder3D_anisotropicKernels_sagittal(xOriginal)  

        # F_UMP_final in "W" (sagittal)
        x_amp = self.anisotropicMaxpool(x)
        x_amp = torch.squeeze(x_amp,-3) 
        
        # 2D CNN (single CB2D)
        x_after_cnn = self.maxpool(self.relu(self.conv(x_amp))) 
        x_ap = torch.squeeze(self.ap(x_after_cnn)) 
        x_clf = self.fc(x_ap)

        return x_clf, x_amp


########################################################################################################################
""" LVO presence detection. AXIAL projection """
########################################################################################################################
""" UTILS """
class conv_block_3d(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1, padding=1):
        super(conv_block_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation_rate),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation_rate),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)    
        
        return x
    

class encoder3D_isotropicKernels_with_symmetry(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super(encoder3D_isotropicKernels_with_symmetry, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=(3,3,3))
        self.Conv1 = conv_block_3d(in_channels=input_channels, out_channels=output_channels[0])
        self.Conv2 = conv_block_3d(in_channels=output_channels[0], out_channels=output_channels[1])
        self.Conv3 = conv_block_3d(in_channels=output_channels[1], out_channels=output_channels[2])
        
    def forward(self, x):
        
        x_flipped = torch.flip(x,[-3])

        """ Encoding path for x """
        x1 = self.Conv1(x) 

        x2 = self.Maxpool(x1) 
        x2 = self.Conv2(x2) 
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) 


        """ Encoding path for x_flipped """
        x1_flipped = self.Conv1(x_flipped) 

        x2_flipped = self.Maxpool(x1_flipped) 
        x2_flipped = self.Conv2(x2_flipped) 
        
        x3_flipped = self.Maxpool(x2_flipped)
        x3_flipped = self.Conv3(x3_flipped) 


        return torch.abs(x1 - x1_flipped), torch.abs(x2 - x2_flipped), torch.abs(x3 - x3_flipped) 



""" 3D to 2D """
# L3P-isotropic with symmetry + 2D CNN (single CB2D)
class L3P_isotropic_with_symmetry_2D_CNN_axial(pl.LightningModule):
    def __init__(self,
                 input_channels= 1,
                 num_channels=[8,16,32],
                 n_classes=1):

        super(L3P_isotropic_with_symmetry_2D_CNN_axial, self).__init__()

        self.encoder3D_isotropicKernels_with_symmetry = encoder3D_isotropicKernels_with_symmetry(input_channels=input_channels, output_channels=[num_channels[0], num_channels[1], num_channels[2]])

        self.anisotropicMaxpool_z = nn.MaxPool3d(kernel_size=(1,1,14),stride=(1,1,14))

        self.conv = nn.Conv2d(num_channels[2], num_channels[2], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(num_channels[2], n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder isotropic with symmetry
        x1, x2, x = self.encoder3D_isotropicKernels_with_symmetry(xOriginal)
        
        # F_UMP_final in "D" (axial)
        x_amp = self.anisotropicMaxpool_z(x)
        x_amp = torch.squeeze(x_amp,-1) 

        # 2D CNN (single CB2D)
        x_after_cnn = self.maxpool(self.relu(self.conv(x_amp))) 
        x_ap = torch.squeeze(self.ap(x_after_cnn)) 

        x_clf = self.fc(x_ap)

        return x_clf, x_amp
