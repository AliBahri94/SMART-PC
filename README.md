<h1>SMART-PC - ICML 2025 🎉 </h1>
<h3>Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds</h3>

📄 [Paper Link on arXiv](https://arxiv.org/pdf/2503.04953)

## Abstract

Test Time Training has emerged as a promising solution to address distribution shifts in 3D point cloud classification. However, existing methods often rely on computationally expensive backpropagation during adaptation, limiting their applicability in real-world, time-sensitive scenarios. In this paper, we introduce SMART-PC, a skeleton-based framework that enhances resilience to corruptions by leveraging the geometric structure of 3D point clouds. During pre-training, our method predicts skeletal representations, enabling the model to extract robust and meaningful geometric features that are less sensitive to corruptions, thereby improving adaptability to test-time distribution shifts.
Unlike prior approaches, SMART-PC achieves real-time adaptation by eliminating backpropagation and updating only BatchNorm statistics, resulting in a lightweight and efficient framework capable of achieving high frame-per-second rates while maintaining superior classification performance. Extensive experiments on benchmark datasets, including ModelNet40-C, ShapeNet-C, and ScanObjectNN-C, demonstrate that SMART-PC achieves state-of-the-art results, outperforming existing methods such as MATE in terms of both accuracy and computational efficiency.


## Overview

<div  align="center">    
 <img src="./figures/method.png" width = "888"  align=center />
</div>




# Preparation

## Requirements
```
PyTorch >= 1.7.0 < 1.11.0  
python >= 3.7  
CUDA >= 9.0  
GCC >= 4.9  
```
To install all additional requirements (open command line and run):
```
pip install -r requirements.txt

cd ./extensions/chamfer_dist
python setup.py install --user

cd ..

cd ./extensions/emd
python setup.py install --user
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Data Preparation
Our code currently supports three different datasets: [ModelNet40](https://arxiv.org/abs/1406.5670), [ShapeNetCore](https://arxiv.org/abs/1512.03012) and [ScanObjectNN](https://arxiv.org/abs/1908.04616).

### First Method: Create Corrupted Datasets 
For this work, you can use the code available in the [MATE GitHub repository](https://github.com/jmiemirza/MATE/tree/master)
  

### Second  Method: Download Corrupted Datasets 
You can download the corrupted datasets from the following [Google Drive link](https://drive.google.com/drive/folders/1v2VP-K0x0TIsPjpmJox6j-CgVPMLhe6Q?usp=sharing).


## Obtaining Pre-Trained Models
All our pretrained models are available at 
this [Google-Drive](https://drive.google.com/drive/folders/15Vf-6_tFQ44PXI1KetDGGzIRwNzfB32P?usp=sharing).

For Org-SO and MATE*, we used this [Google-Drive](https://drive.google.com/drive/folders/1TR46XXp63rtKxH5ufdbfI-X0ZXx8MyKm?usp=share_link).


## Test-Time-Training (TTT)
### Setting data paths 
For TTT, go to `cfgs/tta/tta_<dataset_name>.yaml` and set the `tta_dataset_path` variable to the relative path of the dataset parent directory.  
E.g. if your data for ModelNet-C is in `./data/tta_datasets/modelnet-c`, set the variable to `./data/tta_datasets`.  

<p><strong><span style="color:#2a9d8f;">Online Mode:</span></strong> 
Make sure <code>disable_bn_adaptation</code> is set to <code><strong>False</strong></code> in the config files.</p>

```
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name <dataset_name> --online --grad_steps 1 --config cfgs/tta/tta_<dataset_name>.yaml --ckpts <path/to/pretrained/model>
```

<p><strong><span style="color:#2a9d8f;">Standard Mode:</span></strong> 
Make sure <code>disable_bn_adaptation</code> is set to <code><strong>True</strong></code> in the config files.</p>

```
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name <dataset_name> --grad_steps 20 --config cfgs/tta/tta_<dataset_name>.yaml --ckpts <path/to/pretrained/model>
```

## Training Models
### Setting data paths
To train a new model on one of the three datasets, go to `cfgs/dataset_configs/<dataset_name>.yaml` and set the `DATA_PATH` 
variable in the file to the relative path of the dataset folder.  

### Running training scripts
After setting the paths, a model can be jointly trained by
```
CUDA_VISIBLE_DEVICES=0 python train.py --jt --config cfgs/pre_train/pretrain_<dataset_name>.yaml --dataset <dataset_name>
```  
A model for a supervised only baseline can be trained by
```
CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_<dataset_name>.yaml --dataset <dataset_name>
```  
The trained models can then be found in the corresponding `experiments` subfolder.

## Inference

For a basic inference baseline without adaptation, use
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name <dataset_name> --config cfgs/pre_train/pretrain_<dataset_name>.yaml  --ckpts <path/to/pretrained/model> --test_source
```
Scripts for pretraining, testing and test-time training can also be found in `commands.sh`.


## Acknowledgement

This project is based on Point-MAE ([paper](https://arxiv.org/abs/2203.06604), [code](https://github.com/Pang-Yatian/Point-MAE)), MATE ([paper](https://arxiv.org/pdf/2211.11432), [code](https://github.com/jmiemirza/MATE/tree/master)). Thanks for their wonderful works.



