import os
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


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

class encoder3D_isotropicKernels(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super(encoder3D_isotropicKernels, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=(3,3,3))

        self.Conv1 = conv_block_3d(in_channels=input_channels, out_channels=output_channels[0])
        self.Conv2 = conv_block_3d(in_channels=output_channels[0], out_channels=output_channels[1])
        self.Conv3 = conv_block_3d(in_channels=output_channels[1], out_channels=output_channels[2])
        
    def forward(self, x):
        
        """ Encoding path """
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1) 
        x2 = self.Conv2(x2) 
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3) 

        return x1, x2, x3 

class PLM(nn.Module):
    def __init__(self, poolingsize, in_channels, channels):
        super().__init__()
        self.plm = nn.Sequential(   
            nn.Conv3d(in_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[poolingsize, 1, 1])
        )

    def forward(self, x):
        return self.plm(x)
   
class OutConv2d(nn.Module):
    def __init__(self, channels,n_class):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, n_class, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.out_conv(x)

class vggNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(vggNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
        self.maxPooling = nn.MaxPool3d(kernel_size=2, stride=2, padding=0
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        xMp = self.maxPooling(x3)

        return xMp

class encoder_3VGG(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(encoder_3VGG, self).__init__()
        self.encoder_first = vggNet(in_channels, out_channels, stride=stride, padding=padding)
        self.encoder1 = vggNet(out_channels, out_channels, stride=stride, padding=padding)
        self.encoder2 = vggNet(out_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x):
        x = self.encoder_first(x) 
        x = self.encoder1(x)
        x = self.encoder2(x)

        return x

class encoder_x_xFlip(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding='same'):
        super(encoder_x_xFlip, self).__init__()
        self.encoder = encoder_3VGG(in_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x, xFlip):
        x1 = self.encoder(x)
        x2 = self.encoder(xFlip)

        return x1, x2

class encoder_2VGG(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(encoder_2VGG, self).__init__()
        self.encoder_first = vggNet(in_channels, out_channels, stride=stride, padding=padding)
        self.encoder = vggNet(out_channels, out_channels, stride=stride, padding=padding)

    def forward(self, x):
        x = self.encoder_first(x) 
        x = self.encoder(x)

        return x

class merge_process_with_skip_connection_parallel_gavp_two_more_linear(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, padding=2, depth_processing=2):
        super(merge_process_with_skip_connection_parallel_gavp_two_more_linear, self).__init__()
        self.encoder = encoder_2VGG(in_channels*2, out_channels, stride=1, padding=padding)

        self.ap_x = nn.AvgPool3d(kernel_size=(9,22,16))
        self.ap_flip = nn.AvgPool3d(kernel_size=(9,22,16))
        
        self.ap3 = nn.AvgPool3d(kernel_size=(2,5,4))
        self.lin1 = nn.Linear(in_features=24*3, out_features=24*2)
        self.lin2 = nn.Linear(in_features=24*2, out_features=24)
        self.lin3 = nn.Linear(in_features=24, out_features=n_classes)

    def forward(self, x, x_flip, x_to_concatenate):
        ap_x = (self.ap_x(x))
        ap_flip = self.ap_flip(x_flip)
        merged_layer = torch.abs(x - x_flip) 
        merged_layer = torch.cat([x_to_concatenate,merged_layer],dim=1)
        encoding = self.encoder(merged_layer)
        gavp = self.ap3(encoding)

        gavp_concat = torch.cat([gavp, ap_x, ap_flip],dim=1)
        gavp_flat = torch.flatten(gavp_concat,start_dim=1)
        dense1 = self.lin1(gavp_flat)
        dense2 = self.lin2(dense1)
        dense3 = self.lin3(dense2)

        return dense3
    

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

# L3P-isotropic + 2D CNN (single CB2D)
class L3P_isotropic_2D_CNN_sagittal(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 num_channels=[8,16,32],
                 n_classes=1):

        super(L3P_isotropic_2D_CNN_sagittal, self).__init__()

        self.encoder3D_isotropicKernels = encoder3D_isotropicKernels(input_channels=input_channels, output_channels=[num_channels[0], num_channels[1], num_channels[2]])

        self.anisotropicMaxpool_x = nn.MaxPool3d(kernel_size=(16,1,1),stride=(16,1,1))
        self.anisotropicMaxpool_y = nn.MaxPool3d(kernel_size=(1,20,1),stride=(1,20,1))
        self.anisotropicMaxpool_z = nn.MaxPool3d(kernel_size=(1,1,14),stride=(1,1,14))

        self.seq_x = nn.Sequential(
            nn.Conv2d(num_channels[2], num_channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        self.fc = nn.Linear(num_channels[2], n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder isotropic
        x1,x2,x = self.encoder3D_isotropicKernels(xOriginal)  

        # F_UMP_final in "W" (sagittal)
        x_amp = self.anisotropicMaxpool_x(x)
        x_amp = torch.squeeze(x_amp,-3) 

        # 2D CNN (single CB2D)
        x_after_cnn = torch.squeeze(self.seq_x(x_amp))
        x_clf = self.fc(x_after_cnn)

        return x_clf, x_amp

# PLM-based model
class PLM_based_sagittal(nn.Module):
    def __init__(self, in_channels, channels=[32,64,128], n_classes=1):
        super(PLM_based_sagittal, self).__init__()
        
        self.PLM1 = PLM(5, in_channels, channels[0])
        self.PLM2 = PLM(4, channels[0], channels[1])
        self.PLM3 = PLM(4, channels[1], channels[2])

        self.conv1 = OutConv2d(channels[2],channels[1])
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = OutConv2d(channels[1],channels[0])

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(channels[0], n_classes)

    def forward(self, x):
        x = torch.unsqueeze(x[:,0,0,...],1) 

        # Three PLM modules
        x = self.PLM1(x)
        x = self.PLM2(x) 
        feature = self.PLM3(x) 

        # 2D CNN
        feature = torch.squeeze(feature, -3) 
        conv1 = self.maxpool(self.relu(self.conv1(feature))) 
        conv2 = self.conv2(conv1) 
        
        x_ap = torch.squeeze(self.ap(conv2)) 
        x_clf = self.fc(x_ap)

        return x_clf, feature


""" 2D """
# 2D CNN (single CB2D)
class CNN_2D_single_CB2D(pl.LightningModule):

    def __init__(self,
                 input_channels=1,
                 conv_filters=32,
                 n_classes=1):

        super(CNN_2D_single_CB2D, self).__init__()

        self.conv = nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(conv_filters, n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1)

        # Single CB2D
        x_after_cnn = self.maxpool(self.relu(self.conv(xOriginal)))

        # Global Average Pooling 
        x_ap = torch.squeeze(self.ap(x_after_cnn))

        # Fully connected layer
        x_clf = self.fc(x_ap) 

        return x_clf, None

# 2D CNN (three CB2D)
class CNN_2D_three_CB2D(pl.LightningModule):

    def __init__(self,
                 input_channels=1,
                 conv_filters=32,
                 n_classes=1):
        super(CNN_2D_three_CB2D, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(conv_filters, n_classes)


    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1)

        # 1st CB2D
        x = self.conv1(xOriginal)
        x = self.relu(x)
        x = self.maxpool(x)
 
        # 2nd CB2D
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # 3rd CB2D
        x = self.conv3(x)
        x = self.relu3(x)

        xap = self.ap(x)
        xap = xap.view(xap.size(0), -1)

        x_clf = self.fc(xap)

        return x_clf, None


""" 3D """
# DeepSymNet-v3-based model
class DeepSymNetv3_with_skip_connection_parallel_gavp_two_more_linear(pl.LightningModule):
    def __init__(self, 
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1):
        
        super(DeepSymNetv3_with_skip_connection_parallel_gavp_two_more_linear, self).__init__()
        
       
        self.encoder = encoder_x_xFlip(input_channels, out_channels=n_filters_vgg, stride=1, padding=1)
        self.merge_proc = merge_process_with_skip_connection_parallel_gavp_two_more_linear(in_channels=n_filters_vgg, out_channels=n_filters_vgg, n_classes=number_classes, padding=1, depth_processing=2)
        

    def forward(self, x): 
        # Original and flipped inputs
        xOriginal = torch.unsqueeze(x[:,0,0,...],1)
        xFlip = torch.unsqueeze(x[:,0,1,...],1)        
 
        # Encoding path 
        xEnc, xFlipEnc = self.encoder(xOriginal, xFlip) 
        xMerged = self.merge_proc(xEnc, xFlipEnc, xEnc)

        return xMerged, None





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
    
class encoder3D_anisotropicKernels_with_symmetry(pl.LightningModule):
    def __init__(self, input_channels, output_channels):
        super(encoder3D_anisotropicKernels_with_symmetry, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=(1,1,3))

        self.Conv1 = conv_block_3d_anisotropic(in_channels=input_channels, out_channels=output_channels[0])
        self.Conv2 = conv_block_3d_anisotropic(in_channels=output_channels[0], out_channels=output_channels[1])
        self.Conv3 = conv_block_3d_anisotropic(in_channels=output_channels[1], out_channels=output_channels[2])
        
    def forward(self, x):
          
        x_flipped = torch.flip(x,[-3])

        # Encoding path for original input
        x1 = self.Conv1(x) 

        x2 = self.Maxpool(x1) 
        x2 = self.Conv2(x2) 
        
        x3 = self.Maxpool(x2) 
        x3 = self.Conv3(x3) 

        # Encoding path for flipped input
        x1_flipped = self.Conv1(x_flipped) 

        x2_flipped = self.Maxpool(x1_flipped) 
        x2_flipped = self.Conv2(x2_flipped) 
        
        x3_flipped = self.Maxpool(x2_flipped) 
        x3_flipped = self.Conv3(x3_flipped)

        # L1 difference
        return torch.abs(x3 - x3_flipped) 

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

class merge_process(nn.Module):
    def __init__(self, in_channels, out_channels, n_classes, padding=2, depth_processing=2):
        super(merge_process, self).__init__()
        self.encoder = encoder_2VGG(in_channels, out_channels, stride=1, padding=padding)

        self.act = nn.LeakyReLU(inplace=True)
        self.nclasses = n_classes

        self.ap3 = nn.AvgPool3d(kernel_size=(2,5,4))
        self.lin = nn.Linear(in_features=24, out_features=n_classes)

    def forward(self, x, x_flip):
        merged_layer = torch.abs(x - x_flip) 
        encoding = self.encoder(merged_layer)
        gavgp = torch.flatten(self.ap3(encoding),start_dim=1)
        dense = self.lin(gavgp)

        return dense
    

""" 3D to 2D """
# L3P + 2D CNN (single CB2D)
class L3P_2D_CNN_axial(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 unet_features=[8,16,32],
                 n_classes=1):

        super(L3P_2D_CNN_axial, self).__init__()

        self.encoder3D_anisotropicKernels = encoder3D_anisotropicKernels(input_channels=input_channels, output_channels=[unet_features[0], unet_features[1], unet_features[2]])

        self.anisotropicMaxpool = nn.MaxPool3d(kernel_size=(1,1,9),stride=(1,1,9))

        self.conv = nn.Conv2d(unet_features[2], unet_features[2], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(unet_features[2], n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder
        x = self.encoder3D_anisotropicKernels(xOriginal)  
        
        # F_UMP_final in "D" (axial)
        x_amp = self.anisotropicMaxpool(x) 
        x_amp = torch.squeeze(x_amp,-1) 
        
        # 2D CNN (single CB2D)
        x_after_cnn = self.maxpool(self.relu(self.conv(x_amp))) 
        x_ap = torch.squeeze(self.ap(x_after_cnn)) 

        x_clf = self.fc(x_ap)

        return x_clf, x_amp

# L3P with symmetry + 2D CNN (single CB2D)
class L3P_with_symmetry_2D_CNN_axial(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 unet_features=[8,16,32],
                 n_classes=1):

        super(L3P_with_symmetry_2D_CNN_axial, self).__init__()

        self.encoder3D_anisotropicKernels_with_symmetry = encoder3D_anisotropicKernels_with_symmetry(input_channels=input_channels, output_channels=[unet_features[0], unet_features[1], unet_features[2]])

        self.anisotropicMaxpool = nn.MaxPool3d(kernel_size=(1,1,9),stride=(1,1,9))

        self.conv = nn.Conv2d(unet_features[2], unet_features[2], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(unet_features[2], n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder with symmetry
        x = self.encoder3D_anisotropicKernels_with_symmetry(xOriginal)  

        # F_UMP_final in "D" (axial)
        x_amp = self.anisotropicMaxpool(x) 
        x_amp = torch.squeeze(x_amp,-1) 

        # 2D CNN (single CB2D)
        x_after_cnn = self.maxpool(self.relu(self.conv(x_amp)))
        x_ap = torch.squeeze(self.ap(x_after_cnn)) 

        x_clf = self.fc(x_ap)

        return x_clf, x_amp

# L3P-isotropic + 2D CNN (single CB2D)
class L3P_isotropic_2D_CNN_axial(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 unet_features=[8,16,32],
                 n_classes=1):

        super(L3P_isotropic_2D_CNN_axial, self).__init__()

        self.encoder3D_isotropicKernels = encoder3D_isotropicKernels(input_channels=input_channels, output_channels=[unet_features[0], unet_features[1], unet_features[2]])

        self.anisotropicMaxpool_z = nn.MaxPool3d(kernel_size=(1,1,14),stride=(1,1,14))

        self.conv = nn.Conv2d(unet_features[2], unet_features[2], kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(unet_features[2], n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder isotropic with symmetry
        x1, x2, x = self.encoder3D_isotropicKernels(xOriginal)  

        # F_UMP_final in "D" (axial)
        x_amp = self.anisotropicMaxpool_z(x)
        x_amp = torch.squeeze(x_amp,-1) 

        # 2D CNN (single CB2D)
        x_after_cnn = self.maxpool(self.relu(self.conv(x_amp))) 
        x_ap = torch.squeeze(self.ap(x_after_cnn))

        x_clf = self.fc(x_ap)

        return x_clf, x_amp

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


""" 2D """
# 2D CNN (seven CB2D)
class CNN_2D_seven_CB2D(pl.LightningModule):

    def __init__(self,
                 input_channels=1,
                 conv_filters=32,
                 n_classes=1):
        super(CNN_2D_seven_CB2D, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1).
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(conv_filters, n_classes)


    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1)

        x_clf = self.fc(self.ap(self.seq(xOriginal)))

        return x_clf, None


""" 3D """
# DeepSymNet-v3-based model
class DeepSymNetv3(pl.LightningModule):
    def __init__(self, 
                 input_channels=1, 
                 n_filters_vgg=24, 
                 number_classes=1):
        
        super(DeepSymNetv3, self).__init__()
        
        self.n_outputs = number_classes

        self.encoder = encoder_x_xFlip(input_channels, out_channels=n_filters_vgg, stride=1, padding=1)
        self.merge_proc = merge_process(in_channels=n_filters_vgg, out_channels=n_filters_vgg, n_classes=number_classes, padding=1, depth_processing=2)

    def forward(self, x): 
        # Original and flipped inputs
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 
        xFlip = torch.unsqueeze(x[:,0,1,...],1)
 
        # Encoding path 
        xEnc, xFlipEnc = self.encoder(xOriginal, xFlip) 
        xMerged = self.merge_proc(xEnc, xFlipEnc)

        return xMerged, None


""" 3D to 2.5D """
# 3-views L3P-isotropic + 2D CNN (single CB2D)
class L3P_isotropic_3views_2D_CNN_axial(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 unet_features=[8,16,32],
                 n_classes=1):

        super(L3P_isotropic_3views_2D_CNN_axial, self).__init__()

        self.encoder3D_isotropicKernels = encoder3D_isotropicKernels(input_channels=input_channels, output_channels=[unet_features[0], unet_features[1], unet_features[2]])

        self.anisotropicMaxpool_x = nn.MaxPool3d(kernel_size=(16,1,1),stride=(16,1,1))
        self.anisotropicMaxpool_y = nn.MaxPool3d(kernel_size=(1,20,1),stride=(1,20,1))
        self.anisotropicMaxpool_z = nn.MaxPool3d(kernel_size=(1,1,14),stride=(1,1,14))

        self.seq_x = nn.Sequential(
            nn.Conv2d(unet_features[2], unet_features[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        self.fc = nn.Linear(unet_features[2]*3, n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder isotropic
        x1, x2, x = self.encoder3D_isotropicKernels(xOriginal)  

        # UMP in x
        x_amp_x = self.anisotropicMaxpool_x(x)
        x_amp_x = torch.squeeze(x_amp_x,-3) 

        # UMP in y
        x_amp_y = self.anisotropicMaxpool_y(x) 
        x_amp_y = torch.squeeze(x_amp_y,-2) 

        # UMP in z
        x_amp_z = self.anisotropicMaxpool_z(x) 
        x_amp_z = torch.squeeze(x_amp_z,-1) 

        # 2D CNN (sharing weights) for the three 2D inputs 
        x_after_cnn_x = torch.squeeze(self.seq_x(x_amp_x))
        x_after_cnn_y = torch.squeeze(self.seq_x(x_amp_y))
        x_after_cnn_z = torch.squeeze(self.seq_x(x_amp_z))

        # Concatenation
        x_after_cnn = torch.concatenate([x_after_cnn_x,x_after_cnn_y,x_after_cnn_z],axis=-1)

        x_clf = self.fc(x_after_cnn)

        return x_clf, [x_amp_x, x_amp_y, x_amp_z]

# 3-views L3P-isotropic with symmetry + 2D CNN (single CB2D)
class L3P_isotropic_with_symmetry_3views_2D_CNN_axial(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 unet_features=[8,16,32],
                 n_classes=1):

        super(L3P_isotropic_with_symmetry_3views_2D_CNN_axial, self).__init__()

        self.encoder3D_isotropicKernels = encoder3D_isotropicKernels_with_symmetry(input_channels=input_channels, output_channels=[unet_features[0], unet_features[1], unet_features[2]])

        self.anisotropicMaxpool_x = nn.MaxPool3d(kernel_size=(16,1,1),stride=(16,1,1))
        self.anisotropicMaxpool_y = nn.MaxPool3d(kernel_size=(1,20,1),stride=(1,20,1))
        self.anisotropicMaxpool_z = nn.MaxPool3d(kernel_size=(1,1,14),stride=(1,1,14))

        self.seq_x = nn.Sequential(
            nn.Conv2d(unet_features[2], unet_features[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        self.fc = nn.Linear(unet_features[2]*3, n_classes)

    def forward(self, x):
        xOriginal = torch.unsqueeze(x[:,0,0,...],1) 

        # 3D encoder isotropic
        x1, x2, x = self.encoder3D_isotropicKernels(xOriginal)  

        # UMP in x
        x_amp_x = self.anisotropicMaxpool_x(x)
        x_amp_x = torch.squeeze(x_amp_x,-3) 

        # UMP in y
        x_amp_y = self.anisotropicMaxpool_y(x) 
        x_amp_y = torch.squeeze(x_amp_y,-2) 

        # UMP in z
        x_amp_z = self.anisotropicMaxpool_z(x) 
        x_amp_z = torch.squeeze(x_amp_z,-1) 

        # 2D CNN (sharing weights) for the three 2D inputs 
        x_after_cnn_x = torch.squeeze(self.seq_x(x_amp_x))
        x_after_cnn_y = torch.squeeze(self.seq_x(x_amp_y))
        x_after_cnn_z = torch.squeeze(self.seq_x(x_amp_z))

        # Concatenation
        x_after_cnn = torch.concatenate([x_after_cnn_x,x_after_cnn_y,x_after_cnn_z],axis=-1)

        x_clf = self.fc(x_after_cnn)

        return x_clf, [x_amp_x, x_amp_y, x_amp_z]


""" 2.5D """
# 3-views 2D MIP + 2D CNN (single CB2D)
class CNN_3views_single_CB2D(pl.LightningModule):

    def __init__(self,
                 input_channels=1,
                 conv_filters=32,
                 n_classes=1):

        super(CNN_3views_single_CB2D, self).__init__()

        self.conv = nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ap = nn.AdaptiveAvgPool2d(output_size=1)

        self.fc = nn.Linear(conv_filters*3, n_classes)

    def forward(self, x):
        xOriginalX = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-3),-3),1).type(torch.float32) 
        xOriginalY = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-2),-2),1).type(torch.float32) 
        xOriginalZ = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-1),-1),1).type(torch.float32)

        while len(xOriginalX.shape) > 4:
            xOriginalX = torch.squeeze(xOriginalX,0)
        while len(xOriginalY.shape) > 4:
            xOriginalY = torch.squeeze(xOriginalY,0)
        while len(xOriginalZ.shape) > 4:
            xOriginalZ = torch.squeeze(xOriginalZ,0)
        
        # Single CB2D + Global Average Pooling 
        x_after_cnn = self.maxpool(self.relu(self.conv(xOriginalX)))
        x_apX = torch.squeeze(self.ap(x_after_cnn)) 

        x_after_cnn = self.maxpool(self.relu(self.conv(xOriginalY))) 
        x_apY = torch.squeeze(self.ap(x_after_cnn)) 

        x_after_cnn = self.maxpool(self.relu(self.conv(xOriginalZ))) 
        x_apZ = torch.squeeze(self.ap(x_after_cnn))

        # Concatenation
        x_ap = torch.concatenate([x_apX,x_apY,x_apZ],axis=-1)

        # Fully connected layer
        x_clf = self.fc(x_ap) 

        return x_clf, None

# 3-views 2D MIP + 2D CNN (three CB2D)
class CNN_3views_three_CB2D(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 conv_filters=32,
                 n_classes=1):
        super(CNN_3views_three_CB2D, self).__init__()

        self.three_CB2D = nn.Sequential(
            nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(output_size=1)
        )
                
        self.fc = nn.Linear(conv_filters*3, n_classes)

    def forward(self, x):
        xOriginalX = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-3),-3),1).type(torch.float32) 
        xOriginalY = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-2),-2),1).type(torch.float32) 
        xOriginalZ = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-1),-1),1).type(torch.float32)

        while len(xOriginalX.shape) > 4:
            xOriginalX = torch.squeeze(xOriginalX,0)
        while len(xOriginalY.shape) > 4:
            xOriginalY = torch.squeeze(xOriginalY,0)
        while len(xOriginalZ.shape) > 4:
            xOriginalZ = torch.squeeze(xOriginalZ,0)
        
        # Single CB2D + Global Average Pooling         
        x_clfX = self.three_CB2D(xOriginalX) 
        x_clfY = self.three_CB2D(xOriginalY) 
        x_clfZ = self.three_CB2D(xOriginalZ) 

        # Concatenation + Fully connected layer
        x_clf = self.fc(torch.concatenate([x_clfX,x_clfY,x_clfZ],axis=-1))

        return x_clf, None

# 3-views 2D MIP + 2D CNN (seven CB2D)
class CNN_3views_seven_CB2D(pl.LightningModule):

    def __init__(self,
                 input_channels= 1,
                 conv_filters=32,
                 n_classes=1):
        super(CNN_3views_seven_CB2D, self).__init__()

        self.three_CB2D = nn.Sequential(
            nn.Conv2d(input_channels, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                        
            nn.Conv2d(conv_filters, conv_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(output_size=1)
        )
                
        self.fc = nn.Linear(conv_filters*3, n_classes)

    def forward(self, x):
        xOriginalX = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-3),-3),1).type(torch.float32) 
        xOriginalY = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-2),-2),1).type(torch.float32) 
        xOriginalZ = torch.unsqueeze(torch.squeeze(torch.argmax(x[:,0,...],-1),-1),1).type(torch.float32)

        while len(xOriginalX.shape) > 4:
            xOriginalX = torch.squeeze(xOriginalX,0)
        while len(xOriginalY.shape) > 4:
            xOriginalY = torch.squeeze(xOriginalY,0)
        while len(xOriginalZ.shape) > 4:
            xOriginalZ = torch.squeeze(xOriginalZ,0)
        
        # Single CB2D + Global Average Pooling         
        x_clfX = self.three_CB2D(xOriginalX) 
        x_clfY = self.three_CB2D(xOriginalY) 
        x_clfZ = self.three_CB2D(xOriginalZ) 

        # Concatenation + Fully connected layer
        x_clf = self.fc(torch.concatenate([x_clfX,x_clfY,x_clfZ],axis=-1))

        return x_clf, None

