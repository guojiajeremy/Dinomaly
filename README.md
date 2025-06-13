# Dinomaly (CVPR2025)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dinomaly-the-less-is-more-philosophy-in-multi/multi-class-anomaly-detection-on-mvtec-ad)](https://paperswithcode.com/sota/multi-class-anomaly-detection-on-mvtec-ad?p=dinomaly-the-less-is-more-philosophy-in-multi)


PyTorch Implementation of
"Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection".
[paper](https://arxiv.org/abs/2405.14325)

A very strong, simple, and easy to use baseline for UAD, multi-class UAD, and more. Give me a star if you like it!!!

![fig1](https://github.com/user-attachments/assets/0bb2e555-656f-4218-b93b-844b5894e429)


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

## Eval discrepancy of anomaly localization
In our code implementation, we binarize the GT mask using gt.bool() after down-sampling, specifically gt[gt>0]=1. As pointed out in an issue, the previous common practice is to use gt[gt>0.5]=1. 
The difference between these two binarization approaches is that gt[gt>0]=1 may result in anomaly regions being one pixel larger compared to gt[gt>0.5]=1. This difference does not affect image-level performance metrics, but it has a slight impact on pixel-level evaluation metrics. 

We think gt[gt>0]=1 is a more reasonable choice. It can be seen as max pooling, so that in the down-sampled GT map, any position that corresponds to a region containing at least one anomaly pixel in the original map is marked as anomalous. If an anomaly region is extremely small in the original image (say 2 pixels), gt[gt>0.5]=1 will erase it while gt[gt>0]=1 can keep it.

```
@inproceedings{guo2025dinomaly,
  title={Dinomaly: The less is more philosophy in multi-class unsupervised anomaly detection},
  author={Guo, Jia and Lu, Shuai and Zhang, Weihang and Chen, Fang and Li, Huiqi and Liao, Hongen},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20405--20415},
  year={2025}
}

```

