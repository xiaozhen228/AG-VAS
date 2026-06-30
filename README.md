### **AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models (CVPR 2026)**

Zhen Qu, Xian Tao, Xiaoyi Bao, Dingrong Wang, ShiChen Qu, Zhengtao Zhang, Xingang Wang

[Paper link](https://arxiv.org/pdf/2603.01305)

Large multimodal models (LMMs) exhibit strong task generalization capabilities, offering new opportunities for zero-shot visual anomaly segmentation (ZSAS). However, existing LMM-based segmentation approaches still face fundamental limitations: anomaly concepts are inherently abstract and context-dependent, lacking stable visual prototypes, and the weak alignment between high-level semantic embeddings and pixel-level spatial features hinders precise anomaly localization.
To address these challenges, we present AG-VAS (Anchor-Guided Visual Anomaly Segmentation), a new framework that expands the LMM vocabulary with three learnable semantic anchor tokens—[SEG], [NOR], and [ANO], establishing a unified anchor-guided segmentation paradigm. Specifically, [SEG] serves as an absolute semantic anchor that translates abstract anomaly semantics into explicit, spatially grounded visual entities (e.g., holes or scratches), while [NOR] and [ANO] act as relative anchors that model the contextual contrast between normal and abnormal patterns across categories. To further enhance cross-modal alignment, we introduce a Semantic-Pixel Alignment Module (SPAM) that aligns language-level semantic embeddings with high-resolution visual features, along with an Anchor-Guided Mask Decoder (AGMD) that performs anchor-conditioned mask prediction for precise anomaly localization.
In addition, we curate Anomaly-Instruct20K, a large-scale instruction dataset that organizes anomaly knowledge into structured descriptions of appearance, shape, and spatial attributes, facilitating effective learning and integration of the proposed semantic anchors. Extensive experiments on six industrial and medical benchmarks demonstrate that AG-VAS achieves consistent state-of-the-art performance in the zero-shot setting.

## Table of Contents

- [📖 Introduction](#introduction)
- [🔧 Environments](#environments)
- [📊 Data Preparation](#data-preparation)
- [🚀 Run Experiments](#run-experiments)
- [🔗 Citation](#citation)
- [🙏 Acknowledgements](#acknowledgements)
- [📜 License](#license)

## Introduction

**This repository contains source code for AG-VAS implemented with PyTorch （Accepted by CVPR 2026）.** 

## Data Preparation

The overall composition of the dataset, including the training and test sets, is presented as follows. Note that, except for the anomaly_seg and anomaly_instruct_20K datasets, all other subsets require no additional processing and can be downloaded directly from official or third-party sources. We provide the download links for the datasets in the following section.

```
my_dataset/                              # Root directory
│
├── ade20k/                               # Task: sem_seg
│   ├── annotations/
│   │   ├── training/
│   │   └── validation/
│   ├── images/
│   │   ├── training/
│   │   └── validation/
│   ├── objectInfo150.txt
│   └── sceneCategories.txt
│
├── coco/                                 # Task: sem_seg (images for COCOStuff)
│   └── train2017/
│       └── *.jpg
│
├── cocostuff/                            # Task: sem_seg
│   └── train2017/
│       └── *.png
│
├── mapillary/                            # Task: sem_seg
│   ├── config_v1.2.json
│   ├── config_v2.0.json
│   ├── LICENSE
│   ├── README
│   ├── demo.py
│   ├── testing/
│   │   └── images/
│   ├── training/
│   │   ├── images/
│   │   ├── v1.2/
│   │   └── v2.0/
│   └── validation/
│       ├── images/
│       ├── v1.2/
│       └── v2.0/
│
├── vlpart/                               # Task: sem_seg (part detection)
│   ├── paco/
│   │   └── annotations/
│   │       └── paco_lvis_v1_train.json
│   └── pascal_part/
│       ├── Annotations_Part/
│       ├── train.json
│       ├── train_base.json
│       ├── train_base_one.json
│       ├── train_one.json
│       ├── val.json
│       └── VOCdevkit/
│
├── refer_seg/                            # Task: refer_seg
│   ├── images/
│   │   ├── mscoco/
│   │   │   └── images/
│   │   │       └── train2014/
│   │   └── saiapr_tc-12/
│   ├── refclef/
│   │   ├── instances.json
│   │   ├── refs(unc).p
│   │   └── refs(berkeley).p
│   ├── refcoco/
│   │   ├── instances.json
│   │   ├── refs(unc).p
│   │   └── refs(google).p
│   ├── refcoco+/
│   │   ├── instances.json
│   │   └── refs(unc).p
│   └── refcocog/
│       ├── instances.json
│       ├── refs(umd).p
│       └── refs(google).p
│
├── vqa_dataset/                        # Task: vqa
│   └── llava_dataset
│       ├── llava_instruct_150k.json
│   └── anomalyov_dataset
│       ├── images
│             ├── bmad
│             ├── webAD
│       ├── bmad_zero_shot.json
│       ├── webad_processed.json
│
├── reason_seg/                           # Task: reason_seg
│   └── ReasonSeg/
│       ├── explanatory/
│       │   └── train.json
│       ├── train/
│       │   ├── *.jpg
│       │   └── *.json
│       ├── val/
│       └── test/
├── anomaly_seg/   # Task: anomaly segmentation and reasoning
│   └── mvtec/
│   └── KSDD2/
│   └── RSDD/
│   └── ISIC/
│   └── ColonDB/
│   └── Road/                         
│   └── ZJU/ 
│   └── RealIADC1/ 
│   └── DTD/
│   └── Goods/
│   └── MIAD/
│   └── meta_mvtec.json
│   └── meta_visa.json
│   └── meta_KSDD2.json
│   └── meta_RSDD.json
│   └── meta_ISIC.json
│   └── meta_ColonDB.json
│   └── meta_Road.json                     
│   └── meta_ZJU.json
│   └── meta_RealIADC1.json
│   └── meta_DTD.json
│   └── meta_MIAD.json
│   └── meta_Goods.json
├── anomaly_instruct_20K/
│   └── RealIADC.json 
│   └── DTD.json
│   └── Goods.json  
├────────────────────
```

ade20k: [Download](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)

coco: [Download](http://images.cocodataset.org/zips/train2017.zip)

cocostuff: [Download](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)

mapillary: [Download](https://www.mapillary.com/dataset/vistas)

vlpart: [Download of paco](https://github.com/facebookresearch/paco/tree/main#dataset-setup), [Download of pascal_part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part) 

refer_seg: [images/mscoco](http://images.cocodataset.org/zips/train2014.zip),  [images/saiapr_tc-12](https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip),  [refCOCO](https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip), [refCOCO+](https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip), [refCOCOg](https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip), [refCLEF](https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip) 

vqa_dataset: [LLaVA-Instruct-150k](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json), [anomalyov_dataset](https://xujiacong.github.io/Anomaly-OV/download.txt)

reason_seg: [ReasonSeg](https://github.com/dvlab-research/LISA#dataset)

---

**Prepare the anomaly_seg sub-datasets:**

The organization and construction of the anomaly_seg dataset follow the same procedure as that used in [Bayes-PL](https://github.com/xiaozhen228/Bayes-PFL) and [DictAS](https://github.com/xiaozhen228/DictAS):

> **1、Download and prepare the original [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad),  [VisA](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar),  [RealIAD](https://realiad4ad.github.io/Real-IAD/), [GoodsAD](https://github.com/jianzhang96/GoodsAD), [DTD-Synthetic](https://drive.google.com/file/d/1em51XXz5_aBNRJlJxxv3-Ed1dO9H3QgS/view?usp=drive_link), [MIAD](https://miad-2022.github.io/), [Zju-leaper](http://www.qaas.zju.edu.cn/zju-leaper/) to any desired path. Taking MVTec-AD and VisA as examples, the original dataset format is as follows:**

```
path1
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000.png

path2
├── visa
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
    ├── split_csv
        ├── 1cls.csv
        ├── 1cls.xlsx
```

> **2、Standardize the MVTec-AD and VisA datasets to the same format and generate the corresponding .json files.**

- run **./dataset/make_dataset.py** to generate standardized datasets **./dataset/mvisa/data/visa** and **./dataset/mvisa/data/mvtec**
- run **./dataset/make_meta.py** to generate **./dataset/mvisa/data/meta_visa.json** and **./dataset/mvisa/data/meta_mvtec.json** (This step can be skipped since we have already generated them.)

The format of the standardized datasets is as follows:

```
./datasets/mvisa/data
├── visa
    ├── candle
        ├── train
            ├── good
                ├── visa_0000_000502.bmp
        ├── test
            ├── good
                ├── visa_0011_000934.bmp
            ├── anomaly
                ├── visa_000_001000.bmp
        ├── ground_truth
            ├── anomaly1
                ├── visa_000_001000.png
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── mvtec_000000.bmp
        ├── test
            ├── good
                ├── mvtec_good_000272.bmp
            ├── anomaly
                ├── mvtec_broken_large_000209.bmp
        ├── ground_truth
            ├── anomaly
                ├── mvtec_broken_large_000209.png

├── meta_mvtec.json
├── meta_visa.json
```

**Other Datasets**

Please download the other datasets from Google Drive: [[DTAT.zip]](https://drive.google.com/file/d/194ZQoO8vBvPZp-1fczphWtb0xzkEpCGa/view?usp=drive_link). The processing methods for the datasets are similar to those for MVTec and VisA; all datasets are standardized into the MVTec format with a corresponding meta.json file. 

Dataset Brief Description:

(1) HeadCT, BrainMRI, Br35H, ISIC, CVC-ColonDB, and CVC-ClinicDB are carefully curated by the [AdaCLIP](https://github.com/caoyunkang/AdaCLIP) project, while Endo and Kvasir are curated by the [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP) project. We sincerely appreciate their excellent work and dedication.

(2) KSDD2, RSDD, and DAGM datasets were post-processed by us, including operations such as random cropping. The original DAGM dataset was designed for weakly supervised defect segmentation, and thus its pixel-level annotations are imprecise elliptical labels. To make it suitable for anomaly segmentation, we manually re-annotated the dataset with precise pixel-level labels.
Moreover, since the test sets of these datasets contain no normal samples, they are not directly suitable for anomaly classification. Therefore, when generating standardized datasets, we randomly selected an equal number of normal samples from the training set to match the number of abnormal samples in the test set for evaluation purposes. Due to the randomness involved, re-running ./dataset/make_dataset.py may result in different selections of normal samples. Therefore, we have also uploaded the version used in our paper for reference: [[DATA_three.zip]](https://drive.google.com/file/d/1JrTddHh2THBytMzv8O5Ru8AiRSJDHXIi/view?usp=drive_link).


## Environments

Create a new conda environment and install required packages.

```
conda create -n llava_ov python=3.10
conda activate llava_ov
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda env create -f environment.yml
```

## Acknowledgements

We thank the great works [WinCLIP(zqhang)](https://github.com/zqhang/Accurate-WinCLIP-pytorch),  [WinCLIP(caoyunkang)](https://github.com/caoyunkang/WinClip), [RegAD](https://github.com/MediaBrain-SJTU/RegAD), [VCP-CLIP](https://github.com/xiaozhen228/VCP-CLIP),  [APRIL-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN), [Bayes-PFL](https://github.com/xiaozhen228/Bayes-PFL), [FastRecon](https://github.com/FzJun26th/FastRecon), [AnomalyGPT](https://github.com/CASIA-IVA-Lab/AnomalyGPT), [PromptAD](https://github.com/FuNz-0/PromptAD), [MetaUAS](https://github.com/gaobb/MetaUAS) and [ResAD](https://github.com/xcyao00/ResAD) for assisting with our work.

## Todo list

- Release our AG-VAS paper
- Release our training code of DictAS
- Release our model weights
- Release our training code of DictAS

---

## License

The code and dataset in this repository are licensed under the [MIT license](https://mit-license.org/).