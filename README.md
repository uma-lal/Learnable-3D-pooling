# Learnable 3D pooling (L3P)

This is the supporting code for the paper "Encoding 3D information in 2D feature maps for brain CT-Angiography" (currently under review).

In this work, we introduce learnable 3D pooling (L3P), a convolutional neural network (CNN)-based module designed to compress 3D information into multiple 2D feature maps with end-to-end learnt weights. L3P integrates a lightweight 3D encoder with anisotropic convolutions followed by unidirectional max pooling operations. Our method, which consists of L3P followed by a 2D network that processes the L3P 2D output to generate predictions, is applied to 3D brain CT-Angiography (CTA) in the context of acute ischemic stroke (AIS), specifically, AIS due to large vessel occlusion (LVO). To show the benefits of our work, we designed two different experiments. First, we designed an experiment to classify the LVO-affected brain hemisphere into left or right, which allowed us to analyze the ability of L3P to encode the 3D location in a 2D feature map where the location information was in the axis compressed by the 3D-to-2D transformation. Then, we evaluated it on LVO detection, as a binary classification into LVO presence or LVO absence. We compared the performance of our L3P-based approach to that of baseline 2D and stroke-specific 3D models. L3P-based models achieved results equivalent to those of stroke-specific 3D models while requiring fewer parameters and computational resources and providing better results than 2D models using maximum intensity projection images as input. Our L3P-based models trained with 5-fold cross-validation achieved a mean ROC-AUC of 0.96 for LVO-affected hemisphere detection and 0.91 for LVO detection. Additionally, L3P-based models generated more interpretable feature maps, simplifying the visualization and interpretation of 3D data.

## Requirements

Ubuntu 22.04.4 LTS, python 3.9, pytorch 1.13.1, pytorch-lightning 2.0.2, matplotlib 3.7.1, monai 1.1.0, nibabel 5.1.0, numpy 1.23.5.

## Models

In *models/models.py*, there is the code for all the models used in the paper, including L3P-based models, PLM-based models [1], 2D and 2.5D models and 3D models.

In *models/pretrained weights/*, there are the pretrained weights for the best performing model for the LVO-affected hemisphere detection task (*L**VO-hemisphere-L3P_2D_CNN_sagittal.ckpt*) and for the best performing model for the LVO detection task (*L**VO-presence-L3P_isotropic_with_symmetry_2D_CNN.ckpt*).

## Inference

An example of how to obtain the predictions for the two tasks and visualize the L3P 2D output from 3 example images, run the *inference.ipynb* notebok.

## References

[1] Mingchao Li, Yerui Chen, Zexuan Ji, Keren Xie, Songtao Yuan, Qiang, Chen, and Shuo Li,“Image projection network: 3d to 2d image segmentation in octa images,” *IEEE Transactions on Medical Imaging*, vol. 39, no. 11, pp. 3343–3354, 2020
