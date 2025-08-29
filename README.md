<div align="center">

# Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection

### CVPR 2025 
[![arXiv](https://img.shields.io/badge/arXiv-2405.14325-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2405.14325) [![CVF](https://img.shields.io/badge/CVPR-Paper-b4c7e7.svg?style=plastic)]([https://arxiv.org/abs/2405.14325](https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Dinomaly_The_Less_Is_More_Philosophy_in_Multi-Class_Unsupervised_Anomaly_CVPR_2025_paper.pdf))

</div>

PyTorch Implementation of CVPR 2025
"Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection". The first multi-class UAD model that can compete with single-class SOTAs !!!

**Give me a ⭐️ if you like it.**

![fig1](https://github.com/user-attachments/assets/0bb2e555-656f-4218-b93b-844b5894e429)


## News
 - 05.2024: Arxiv preprint and github code released🚀
 
 - 09.2024: Rejected by NeurIPS 2024 with 5 positive scores and no negative score, because "AC: lack of novelty"😭. Wish me good luck.
 
 - 02.2025: Accepted by CVPR 2025🎉
 
 - 07.2025: Spoil alert: We will come back with Dinomly-2😛

 - 07.2025: Dinomaly has been integrated in Intel open-edge [Anomalib](https://github.com/open-edge-platform/anomalib) in v2.1.0. Great thanks to the contributors for the nice reproduction and integration. Anomalib is a comprehensive library for benchmarking, developing and deploying deep learning anomaly detection algorithms.

 - 08.2025: I have sucessfully implement [DINOv3](https://ai.meta.com/dinov3/) on Dinomaly. The pixel-level performance is much better, with slightly lower image-level performance. DINOv3-Large on MVTecAD: I-Auroc:0.9970, P-AUROC:0.9878, P-AP:0.7422, P-F1:0.7184, P-AUPRO:0.9580. Due to DINOv3 requiring newer versions of Python (>3.10) and PyTorch (>2.7), it is not provided in this repository. You can refer to [cnlab](https://github.com/cnulab)'s forked [branch](https://github.com/cnulab/Dinomaly). Great thanks!


## Abstract

Recent studies highlighted a practical setting of unsupervised anomaly detection (UAD) that builds a unified model for multi-class images. Despite various advancements addressing this challenging task, the detection performance under the multi-class setting still lags far behind state-of-the-art class-separated models. Our research aims to bridge this substantial performance gap. In this paper, we introduce a minimalistic reconstruction-based anomaly detection framework, namely Dinomaly, which leverages pure Transformer architectures without relying on complex designs, additional modules, or specialized tricks. Given this powerful framework consisted of only Attentions and MLPs, we found four simple components that are essential to multi-class anomaly detection: (1) Foundation Transformers that extracts universal and discriminative features, (2) Noisy Bottleneck where pre-existing Dropouts do all the noise injection tricks, (3) Linear Attention that naturally cannot focus, and (4) Loose Reconstruction that does not force layer-to-layer and point-by-point reconstruction. Extensive experiments are conducted across popular anomaly detection benchmarks including MVTec-AD, VisA, and Real-IAD. Our proposed Dinomaly achieves impressive image-level AUROC of 99.6%, 98.7%, and 89.3% on the three datasets respectively (99.8%, 98.9%, 90.1% with ViT-L), which is not only superior to state-of-the-art multi-class UAD methods, but also achieves the most advanced class-separated UAD records.

## 1. Environments

Create a new conda environment and install required packages.

```
conda create -n my_env python=3.8.12
conda activate my_env
pip install -r requirements.txt
```
Experiments are conducted on NVIDIA GeForce RTX 3090 (24GB). Same GPU and package version are recommended. 

## 2. Prepare Datasets
Noted that `../` is the upper directory of Dinomaly code. It is where we keep all the datasets by default.
You can also alter it according to your need, just remember to modify the `data_path` in the code. 

### MVTec AD

Download the MVTec-AD dataset from [URL](https://www.mvtec.com/company/research/datasets/mvtec-ad).
Unzip the file to `../mvtec_anomaly_detection`.
```
|-- mvtec_anomaly_detection
    |-- bottle
    |-- cable
    |-- capsule
    |-- ....
```


### VisA

Download the VisA dataset from [URL](https://github.com/amazon-science/spot-diff).
Unzip the file to `../VisA/`. Preprocess the dataset to `../VisA_pytorch/` in 1-class mode by their official splitting 
[code](https://github.com/amazon-science/spot-diff).

You can also run the following command for preprocess, which is the same to their official code.

```
python ./prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file ./prepare_data/split_csv/1cls.csv
```
`../VisA_pytorch` will be like:
```
|-- VisA_pytorch
    |-- 1cls
        |-- candle
            |-- ground_truth
            |-- test
                    |-- good
                    |-- bad
            |-- train
                    |-- good
        |-- capsules
        |-- ....
```
 
### Real-IAD
Contact the authors of Real-IAD [URL](https://realiad4ad.github.io/Real-IAD/) to get the net disk link.

Download and unzip `realiad_1024` and `realiad_jsons` in `../Real-IAD`.
`../Real-IAD` will be like:
```
|-- Real-IAD
    |-- realiad_1024
        |-- audiokack
        |-- bottle_cap
        |-- ....
    |-- realiad_jsons
        |-- realiad_jsons
        |-- realiad_jsons_sv
        |-- realiad_jsons_fuiad_0.0
        |-- ....
```

## 3. Run Experiments
Multi-Class Setting
```
python dinomaly_mvtec_uni.py --data_path ../mvtec_anomaly_detection
```
```
python dinomaly_visa_uni.py --data_path ../VisA_pytorch/1cls
```
```
python dinomaly_realiad_uni.py --data_path ../Real-IAD
```

Conventional Class-Separted Setting
```
python dinomaly_mvtec_sep.py --data_path ../mvtec_anomaly_detection
```
```
python dinomaly_visa_sep.py --data_path ../VisA_pytorch/1cls
```
```
python dinomaly_realiad_sep.py --data_path ../Real-IAD
```

Training Unstability: The optimization can be unstable with loss spikes (e.g. ...0.05, 0.04, 0.04, **0.32**, **0.23**, 0.08...)
, which can be harmful to performance. This occurs very very rare. If you see such loss spikes during training, consider change a random seed.

## Results

**A. Compare with MUAD SOTAs:**
<div align="center">

<img alt="image" src="https://github.com/user-attachments/assets/082922bb-e8f8-4efc-9597-2a7dc8577d6e" />

<img width="869" height="482" alt="image" src="https://github.com/user-attachments/assets/9da30ae7-5c7f-4117-ad93-bf12f0fd98f0" />

</div>



**Dinomaly can perfectly scale with model size, input image size, and the choice of foundation model.**

**B. Model Size:**
<div align="center">

<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/6f388ab7-0b81-450b-ae13-358a00c74f3f" />

<img width="865" height="190" alt="image" src="https://github.com/user-attachments/assets/a5d7c83f-bc64-4704-8607-a7a00cffe545" />
<img width="700" alt="image" src="https://github.com/user-attachments/assets/5005caed-2294-4766-92ed-ee93df5c5428" />

</div>


**C. Input Size:**
<div align="center">

<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/e9a324a3-7f26-4d69-8806-a183042a3388" />

<img width="865" height="302" alt="image" src="https://github.com/user-attachments/assets/4f259320-2e4b-4796-aa7e-740bbd246d37" />

</div>

**D. Choice of Foundaiton Model:**
<div align="center">

<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/a1ae0beb-ac5d-4926-94d4-4a99e07de03b" />

<img width="865" height="474" alt="image" src="https://github.com/user-attachments/assets/8c95f29b-578e-481d-bf0c-75429f76158f" />

</div>


## Eval discrepancy of anomaly localization
In our code implementation, we binarize the GT mask using gt.bool() after down-sampling, specifically gt[gt>0]=1. As pointed out in an issue, the previous common practice is to use gt[gt>0.5]=1. 
The difference between these two binarization approaches is that gt[gt>0]=1 may result in anomaly regions being one pixel larger compared to gt[gt>0.5]=1. This difference does not affect image-level performance metrics, but it has a slight impact on pixel-level evaluation metrics. 

We think gt[gt>0]=1 is a more reasonable choice. It can be seen as max pooling, so that in the down-sampled GT map, any position that corresponds to a region containing at least one anomaly pixel in the original map is marked as anomalous. If an anomaly region is extremely small in the original image (say 2 pixels), gt[gt>0.5]=1 will erase it while gt[gt>0]=1 can keep it.

## Citation
```
@inproceedings{guo2025dinomaly,
  title={Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection},
  author={Guo, Jia and Lu, Shuai and Zhang, Weihang and Chen, Fang and Li, Huiqi and Liao, Hongen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20405--20415},
  year={2025}
}
```

