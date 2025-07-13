# [MICCAI2025] D-CAM: Learning Generalizable Weakly-Supervised Medical Image Segmentation from Domain-invariant CAM

## Introduction
The implementation of **[D-CAM: Learning Generalizable Weakly-Supervised Medical Image Segmentation from Domain-invariant CAM]**.

## Abstract
Weakly-supervised medical image segmentation with only image-level annotation is particularly challenging to infer precise pixel-wise predictions. Existing works are usually highly restricted by the assumption that the medical images for training and testing are under the same distribution. However, a robust weakly-supervised segmentation model needs to show accurate inference on medical images from unseen distributions. Different feature distributions can lead to a dramatic shift in the feature activation and class activation map (CAM), which in turn leads to the degradation of pseudo labels. In this paper, we aim to learn generalizable weakly-supervised medical image segmentation by focusing on enhancing the domain invariance for pseudo labels. A novel domain-invariant CAM learning scheme (D-CAM) is proposed, in which the content and style are decoupled during training. By inferring domain-invariant pseudo labels, the supervision of a segmentation model is more generalizable to different target domains. Extensive experiments under multiple generalized medical image segmentation settings show the state-of-the-art performance of our D-CAM.

## Requirements
- Python 3.7
- pytorch
- torchvision
- CUDA
- 1×GPU

## Usage
### Prerequisite
- To train the stage1 and stage2, you should download the pretrained weight of ResNet-38 and DeepLab V3+ from [baidu cloud](https://pan.baidu.com/s/1sQp4Na-883pSxgMWK4wcRQ) (with code **nylc**) or [onedrive](https://1drv.ms/u/s!AgOtqK2ZncKlgoRobleElpBC5rbf7A?e=bDfqks) and put them in the **init_weights** folder.
- We use LUAD-HistoSeg and BCSS-WSSS for training.

#### 1、Train the classification model: 

```
python 1_train_stage1.py --dataset luad --trainroot datasets/LUAD-HistoSeg/train/ --testroot dataset/LUAD-HistoSeg/test/ --max_epoches 20 --batch_size 20
```
#### 2、Generate pesudo mask: 

```
python 2_generate_PM.py --dataroot datasets/LUAD-HistoSeg --dataset luad --weights checkpoints/stage1_checkpoint_trained_on_luad.pth
```
#### 3、Train the segmentation model: 

```
python 3_train_stage2.py --dataset luad --dataroot datasets/LUAD-HistoSeg --epochs 20 -- Is_GM True --resume_stage1 checkpoints/stage1_checkpoint_trained_on_luad.pth --resume init_weights/deeplab-resnet.pth.tar
```
