![Python >=3.5](https://img.shields.io/badge/Python->=3.5-yellow.svg)
![PyTorch >=1.0](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)

Swin ReID: Swin Transformer-based Object Re-Identification


## Pipeline

![image](https://user-images.githubusercontent.com/34055413/188075291-90bbce2f-79a2-406f-8c84-69b5984c8fbf.png)



## Requirements

### Installation

Please refer to TransReID.

```bash
pip install -r requirements.txt

(we use /torch 1.7.1 /torchvision 0.8.2 /timm 0.3.2 /cuda 11.7 /  for training and evaluation.
Note that we use torch.cuda.amp to accelerate speed of training which requires pytorch >=1.6)
```

### Prepare Datasets

```bash
mkdir data
```

Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775),[Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset), and the vehicle datasets [VehicleID](https://www.pkuml.org/resources/pku-vehicleid.html), [VeRi-776](https://github.com/JDAI-CV/VeRidataset), 
Then unzip them and rename them under the directory like

```
data
├── market1501
│   └── images ..
├── MSMT17
│   └── images ..
├── dukemtmcreid
│   └── images ..
├── Occluded_Duke
│   └── images ..
├── VehicleID_V1.0
│   └── images ..
└── VeRi
    └── images ..
```

### Prepare DeiT or ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth), [ViT-Small](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth), [DeiT-Small](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth), [DeiT-Base](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth)

The ImageNet pretrained swin transformer model:
[Swin-base]https://drive.google.com/file/d/1ESQNMrZUsqqboym5GtKeV9L4-m8dpcAr/view?usp=sharing

## Training

We utilize 1  GPU for training.

```bash
python train.py --config_file configs/transformer_base.yml MODEL.DEVICE_ID "('your device id')" MODEL.STRIDE_SIZE ${1} MODEL.SIE_CAMERA ${2} MODEL.SIE_VIEW ${3} MODEL.JPM ${4} MODEL.TRANSFORMER_TYPE ${5} OUTPUT_DIR ${OUTPUT_DIR} DATASETS.NAMES "('your dataset name')"
```

#### Arguments

- `${1}`: stride size for pure transformer, e.g. [16, 16], [14, 14], [12, 12]
- `${2}`: whether using SIE with camera, True or False.
- `${3}`: whether using SIE with view, True or False.
- `${4}`: whether using JPM, True or False.
- `${5}`: choose transformer type from `'vit_base_patch16_224_TransReID'`,(The structure of the deit is the same as that of the vit, and only need to change the imagenet pretrained model)  `'vit_small_patch16_224_TransReID'`,`'deit_small_patch16_224_TransReID'`,
- `${OUTPUT_DIR}`: folder for saving logs and checkpoints, e.g. `../logs/market1501`

**or you can directly train with following  yml and commands:**

```bash
# DukeMTMC TransReID (baseline + SIE + JPM)
python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC Swin TransReID with stride size [12, 12] (baseline)
python train.py --config_file configs/DukeMTMC/swin_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# MSMT17
python train.py --config_file configs/MSMT17/swin_transreid_stride.yml MODEL.DEVICE_ID "('0')"
# OCC_Duke
python train.py --config_file configs/OCC_Duke/swin_transreid_stride.yml MODEL.DEVICE_ID "('0')"
# Market
python train.py --config_file configs/Market/swin_transreid_stride.yml MODEL.DEVICE_ID "('0')"
```

Tips:  For person datasets  with size 256x128, Swin TransReID with stride occupies 12GB GPU memory.
## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Some examples:**

```bash
# DukeMTMC
python test.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT '../logs/duke_vit_transreid_stride/transformer_120.pth'
# MSMT17
python test.py --config_file configs/MSMT17/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT '../logs/msmt17_vit_transreid_stride/transformer_120.pth'
# OCC_Duke
python test.py --config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT '../logs/occ_duke_vit_transreid_stride/transformer_120.pth'
# Market
python test.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"  TEST.WEIGHT '../logs/market_vit_transreid_stride/transformer_120.pth'

```

## Acknowledgement

Codebase from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) , [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)


If you find this code useful for your research, please cite TransReID.

## Citation
```
@InProceedings{He_2021_ICCV,
    author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
    title     = {TransReID: Transformer-Based Object Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15013-15022}
}
```

