# Learning convolutional multi-level transformers for image-based person re-identification

#########################################################

This is repository contains the code for the paper:

Learning convolutional multi-level transformers for image-based person re-identification

(https://doi.org/10.1007/s44267-023-00025-8)

#########################################################

## Installation
```bash
pip install -r requirements.txt
```
(we use /torch 1.10.1 /torchvision 0.11.2 /timm 0.4.9 for training and evaluation.）

## Prepare Datasets
For using the codes, please download the public Market-1501 and Duke datasets.


### Prepare ResNet-50 Pre-trained Models
Please download the ImageNet pretrained resnet50 model.


## Training and Test
 Run the train.py and test.py with your GPUs. You can get the same results in the above paper.  


## Citation 

if you use this code for your research, please cite our paper:

```
Yan, P., Liu, X., Zhang, P. et al. Learning convolutional multi-level transformers for image-based person re-identification. Vis. Intell. 1, 24 (2023). https://doi.org/10.1007/s44267-023-00025-8

```

## Contact
If you have any problems. Please concat

zhpp@dlut.edu.cn or yanpl@mail.dlut.edu.cn
