# [MICCAI2025] D-CAM: Learning Generalizable Weakly-Supervised Medical Image Segmentation from Domain-invariant CAM

## Introduction
The implementation of **[D-CAM: Learning Generalizable Weakly-Supervised Medical Image Segmentation from Domain-invariant CAM]**.

## Abstract
Weakly-supervised medical image segmentation with only image-level annotation is particularly challenging to infer precise pixel-wise predictions. Existing works are usually highly restricted by the assumption that the medical images for training and testing are under the same distribution. However, a robust weakly-supervised segmentation model needs to show accurate inference on medical images from unseen distributions. Different feature distributions can lead to a dramatic shift in the feature activation and class activation map (CAM), which in turn leads to the degradation of pseudo labels. In this paper, we aim to learn generalizable weakly-supervised medical image segmentation by focusing on enhancing the domain invariance for pseudo labels. A novel domain-invariant CAM learning scheme (D-CAM) is proposed, in which the content and style are decoupled during training. By inferring domain-invariant pseudo labels, the supervision of a segmentation model is more generalizable to different target domains. Extensive experiments under multiple generalized medical image segmentation settings show the state-of-the-art performance of our D-CAM.

## Datasets
We have released both datasets via Google Drive ([LUAD-HistoSeg](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr?usp=sharing) and [BCSS-WSSS](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77?usp=sharing).).
We would like to thank Amgad et al. for this excellent dataset. The original BCSS dataset can be download at this link [BCSS-link](https://github.com/PathologyDataScience/CrowdsourcingDataset-Amgadetal2019).

## Checkpoints
Download our checkpoints of the final models trained on two datasets from the [Google Drive](https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing) and [baidu cloud](https://pan.baidu.com/s/1YWGw8UKwtPkx6E00XP3JTQ) (with code **on7z**):

- [stage1_checkpoint_trained_on_luad.pth](https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing)
- [stage2_checkpoint_trained_on_luad.pth](https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing)
- [stage1_checkpoint_trained_on_bcss.pth](https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing)
- [stage2_checkpoint_trained_on_bcss.pth](https://drive.google.com/drive/folders/1pRsf-TulOARYN13J-_iJNw_VUa2d1NnZ?usp=sharing)

## Requirements
- Python 3.7
- pytorch
- torchvision
- CUDA
- 1×GPU
- visdom

## Usage
### Prerequisite
- To train the stage1 and stage2, you should download the pretrained weight of ResNet-38 and DeepLab V3+ from [baidu cloud](https://pan.baidu.com/s/1sQp4Na-883pSxgMWK4wcRQ) (with code **nylc**) or [onedrive](https://1drv.ms/u/s!AgOtqK2ZncKlgoRobleElpBC5rbf7A?e=bDfqks) and put them in the **init_weights** folder.
- And you can download our datasets ([LUAD-HistoSeg](https://drive.google.com/drive/folders/1E3Yei3Or3xJXukHIybZAgochxfn6FJpr?usp=sharing) and [BCSS-WSSS](https://drive.google.com/drive/folders/1iS2Z0DsbACqGp7m6VDJbAcgzeXNEFr77?usp=sharing)) and unpack them into the **datasets** folder.

WSSS-Tissue/

    |_ datasets
    |     |_ BCSS-WSSS/
    |         |_ train/
    |         |_ val/
    |         |_ test/
    |     |_ LUAD-HistoSeg/
    |         |_ train/
    |         |_ val/
    |         |_ test/
 
- Finally, you need to launch "visdom" before running the whole pipline.

### Run the whole pipeline
- In this work, our pipeline include 3 part~(Stage 1, PM generation, Stage2). You can directly run run_pipeline.sh to complete the three-part training and test.
- [Note] To run the whole pipeline, you need to specify the dataset related hyperparameters. Please see the command in run_pipeline.sh.
```
bash run_pipeline.sh
```
### Run each step: 
- The whole pipeline of our model includes the step of classification model training, multi-layer pseudo masks generation, and segmentation model training. You can use following scripts to run each step.
- [Note] Please specify the argument in the command. You can also check run_pipeline.sh to see more details.
#### 1、Train the classification model with the image-level label: 

```
python 1_train_stage1.py --dataset luad --trainroot datasets/LUAD-HistoSeg/train/ --testroot dataset/LUAD-HistoSeg/test/ --max_epoches 20 --batch_size 20
```
#### 2、Generate multi-layers pesudo mask by classification model: 

```
python 2_generate_PM.py --dataroot datasets/LUAD-HistoSeg --dataset luad --weights checkpoints/stage1_checkpoint_trained_on_luad.pth
```
#### 3、Train the segmentation model with multi-layer pesudo masks: 

```
python 3_train_stage2.py --dataset luad --dataroot datasets/LUAD-HistoSeg --epochs 20 -- Is_GM True --resume_stage1 checkpoints/stage1_checkpoint_trained_on_luad.pth --resume init_weights/deeplab-resnet.pth.tar
```
