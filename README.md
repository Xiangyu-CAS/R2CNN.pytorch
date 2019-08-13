# R2CNN in PyTorch 1.2
Pytorch Implementation of "R2CNN Rotational Region CNN for Orientation Robust Scene Text Detection" [paper](https://arxiv.org/abs/1706.09579)
, it is based on facebook's [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Inference on ICDAR 2015 dataset
**1. Download model [R2CNN_R50_ICDAR2015](https://drive.google.com/open?id=1sXI00SG3VdwWWZLP6ZaBxvqph77oackI)**

**2. single image inference**
````
cd ./tools
python inference_engine.py
````

![01](tools/ICDAR2015/img_14.jpg)
![02](tools/ICDAR2015/img_60.jpg)
![03](tools/ICDAR2015/img_108.jpg)

## Perform training on ICDAR2015 dataset
**1. Download [icdar2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) dataset and pretrain model from maskrcnn-bencmark**
````
cd ./tools
mkdir datasets
ln -s PATH_ICDAR2015 datasets/ICDAR2015
mkdir pretrain
cd pretrain
wget https://download.pytorch.org/models/maskrcnn/e2e_faster_rcnn_R_50_FPN_1x.pth
````
**2. Convert annotations to COCO style**
````
cd ./tools/ICDAR2015
python convert_icdar_to_coco.py
````
**3. start training**
````
cd ./tools
python train_net.py 
````

## TODO
- [x] 


## Citations
Please consider citing this project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.
```
@misc{r2cnn,
author = {Yingying Jiang, Xiangyu Zhu, Xiaobing Wang, Shuli Yang, Wei Li, Hua Wang, Pei Fu, Zhenbo Luo},
title = {R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection},
conference = {ICPR2018}
year = {2017},
}
```