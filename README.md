
# Object Detection in Aerial Images [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

🔥 A curated list of awesome resources for generic object detection in aerial images.

:heavy_exclamation_mark: Updated at 2024-06.





--------------------------------------------------------------------------------------
<!--TOC-->

## Contents:
- [Overview](#Overview)
- [Oriented Object Detection](#Oriented-Object-Detection)
- [Instance Segmentation](#Instance-Segmentation)
- [Small Object Detection](#Small-Object-Detection)
- [UAV Object Dectection](#UAV-Object-Detection)
- [Dataset](#Dataset)
- [Appendix](#Appendix)

--------------------------------------------------------------------------------------



## Overview

**No.** | **Year** | **Pub.** | **Title** | **Links**
:-: | :-: | :-:  | :- | :- 
10 | 2023 |  GRSM  | Remote Sensing Object Detection Meets Deep Learning: A Meta-review of Challenges and Advances   <br> <sub><sup>*Xiangrong Zhang, Tianyang Zhang, Guanchun Wang, Peng Zhu, Xu Tang, Xiuping Jia, Licheng Jiao*</sup></sub>  | [Paper](https://arxiv.org/abs/2309.06751)/Code
09 | 2023 |  PAMI  | Towards Large-Scale Small Object Detection: Survey and Benchmarks <br> <sub><sup>*Gong Cheng, Xiang Yuan, Xiwen Yao, Kebing Yan, Qinghua Zeng, Xingxing Xie, Junwei Han*</sup></sub>  | [Paper](https://arxiv.org/abs/2207.14096)/[Data](https://shaunyuan22.github.io/SODA/)
08 | 2023 |  arXiv | Oriented Object Detection in Optical Remote Sensing Images using Deep Learning: A Survey <br> <sub><sup>*Kun Wang, Zi Wang, Zhang Li, Ang Su, Xichao Teng, Minhao Liu, Qifeng Yu*</sup></sub>  | [Paper](https://arxiv.org/abs/2302.10473)/Code 
07 | 2021 | **PAMI** | <span style="white-space:nowrap;">Object Detection in Aerial Images: A Large-Scale Benchmark and Challenges `DOTA` &emsp;&emsp;</span>  <br><sub><sup>*Jian Ding, Nan Xue, Gui-Song Xia, Xiang Bai, et al.*</sup></sub> | [Paper](https://arxiv.org/pdf/2102.12219.pdf)/[Proj](https://captain-whu.github.io/DOTA/) 
06 | 2020 | IJCV | Deep Learning for Generic Object Detection: A Survey `GOD`  <br><sub><sup>*Li Liu, Wanli Ouyang, et al.*</sup></sub>  | [Paper](https://link.springer.com/article/10.1007/s11263-019-01247-4#Abs1)/Code 
05 | 2020 | JPRS | Object detection in optical remote sensing images: A survey and a new benchmark `HBB`  <br><sub><sup>*Ke Li, Gong Cheng, et al.*</sup></sub>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271619302825)/[Data](https://gcheng-nwpu.github.io/#Datasets)  
04 | 2019 | TNNLS | Object Detection With Deep Learning: A Review `GOD`   <br><sub><sup>*Zhongqiu Zhao, et al.*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/document/8627998)/Code 
03 | 2018 | JPRS | A review of accuracy assessment for object-based image analysis: From per-pixel to per-polygon approaches  <br><sub><sup>*Su Ye,  Rahul Rakshit, et al.*</sup></sub>   | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271618300947)/Code
02 | 2018 | **CVPR** | DOTA: A Large-Scale Dataset for Object Detection in Aerial Images `DOTA`    <br><sub><sup>*Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.pdf)/[Code](https://github.com/dingjiansw101/AerialDetection) 
01 | 2016 | JPRS | A survey on object detection in optical remote sensing images   <br><sub><sup>*Gong Cheng, Junwei Han*</sup></sub>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271616300144)/Code



## Oriented Object Detection

<!-- Dataset: DOTA, HRSC2016, ICDAR2015, ICDAR2017 MLT, MSRA-TD500, UCAS-AOD, FDDB, OHD-SJTU, SSDD++, Total-Text. -->

#### Preprint

**Year** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :- 
2024 | arXiv | MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining   <br><sub><sup>*Di Wang, Jing Zhang, Minqiang Xu, Lin Liu, Dongsheng Wang, Erzhong Gao, Chengxi Han, Haonan Guo, Bo Du, Dacheng Tao, Liangpei Zhang*</sup></sub> | [Paper](https://arxiv.org/abs/2403.13430)/[Code](https://github.com/ViTAE-Transformer/MTP)
2024 | arXiv | GOOD: Towards Domain Generalized Orientated Object Detection   <br><sub><sup>*Qi Bi, Beichen Zhou, Jingjun Yi, Wei Ji, Haolan Zhan, Gui-Song Xia*</sup></sub> | [Paper](https://arxiv.org/abs/2402.12765)/Code
2023  | arXiv | P2RBox: A Single Point is All You Need for Oriented Object Detection   <br><sub><sup>*Guangming Cao, Xuehui Yu, Wenwen Yu, Xumeng Han, Xue Yang, Guorong Li, Jianbin Jiao, Zhenjun Han*</sup></sub> | [Paper](https://arxiv.org/abs/2311.13128)/Code
2022 | arXiv | DAFNe: A One-Stage Anchor-Free Deep Model for Oriented Object Detection <br><sub><sup>*Steven Lang, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2109.06148)/[Code](https://github.com/steven-lang/DAFNe) 



<!-- 
**No.** | **Pub.** | **Title** | **Authors** | **Links**
:-: | :-: | :-  | :- | :-
2022 | WACV | TricubeNet: 2D Kernel-Based Object Representation for Weakly-Occluded Oriented Object Detection | Beomyoung Kim, Janghyeon Lee, et al. | [Paper](https://arxiv.org/abs/2104.11435)/Code
2022 | TGRS | MRDet: A Multi-Head Network for Accurate Oriented Object Detection in Aerial Images | Ran Qin, Yunhong Wang, et al. | [Paper](https://arxiv.org/pdf/2012.13135.pdf)/Code
2021 | TGRS | CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote Sensing Images | Qi Ming, et al. | [Paper](https://arxiv.org/pdf/2101.06849.pdf)/[Code](https://github.com/ming71/CFC-Net)
2021 | TGRS | Align Deep Features for Oriented Object Detection | Jiaming Han, Jian Ding, et al. | [Paper](https://arxiv.org/abs/2008.09397)/[Code](https://github.com/csuhan/s2anet)
2021 | GRSL | Optimization for Oriented Object Detection via Representation Invariance Loss | Qi Ming, Zhiqiang Zhou, et al. | [Paper](https://arxiv.org/abs/2103.11636)/[Code](https://github.com/ming71/RIDet) 
2021 | TGRS | Anchor Retouching via Model Interaction for Robust Object Detection in Aerial Images <br><sub><sup>*Dong Liang, Qixiang Geng, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2112.06701)/[Code](https://github.com/QxGeng/DEA-Net)
2021 | arXiv06 | Oriented Object Detection with Transformer <br><sub><sup>*Teli Ma, et al.*</sup></sub>  | [Paper](https://arxiv.org/pdf/2106.03146.pdf)/Code
-->



#### 2024

**No.** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :-
|   11  | TPAMI  |  Learning to Holistically Detect Bridges from Large-Size VHR Remote Sensing Imagery   <br> <sup><sub>*Yansheng Li; Junwei Luo; Yongjun Zhang; Yihua Tan; Jin-Gang Yu; Song Bai*</sub></sup>  | [Paper](https://arxiv.org/abs/2312.02481)/[Code](https://luo-z13.github.io/GLH-Bridge-page/)
|   10  | CVPR  | Weakly Misalignment-free Adaptive Feature Alignment for UAVs-based Multimodal Object Detection    <br> <sup><sub>*Chen Chen, Jiahao Qi, Xingyue Liu, Kangcheng Bin, Ruigang Fu, Xikun Hu, Ping Zhong*</sub></sup>  | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Weakly_Misalignment-free_Adaptive_Feature_Alignment_for_UAVs-based_Multimodal_Object_Detection_CVPR_2024_paper.html)/Code
|   09  | CVPR  | WildlifeMapper: Aerial Image Analysis for Multi-Species Detection and Identification   <br> <sup><sub>*Satish Kumar, Bowen Zhang, Chandrakanth Gudavalli, Connor Levenson, Lacey Hughey, et al.*</sub></sup>  | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Kumar_WildlifeMapper_Aerial_Image_Analysis_for_Multi-Species_Detection_and_Identification_CVPR_2024_paper.html)/[Code](https://github.com/UCSB-VRL/WildlifeMapper)
|   08  | CVPR  | PointOBB: Learning Oriented Object Detection via Single Point Supervision    <br> <sup><sub>*Junwei Luo, Xue Yang, Yi Yu, Qingyun Li, Junchi Yan, Yansheng Li*</sub></sup>  | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_PointOBB_Learning_Oriented_Object_Detection_via_Single_Point_Supervision_CVPR_2024_paper.html)/[Code](https://github.com/Luo-Z13/pointobb) 
|   07  | CVPR	| Rotated Multi-Scale Interaction Network for Referring Remote Sensing Image Segmentation   <br> <sup><sub>*Sihan Liu, Yiwei Ma, Xiaoqing Zhang, Haowei Wang, Jiayi Ji, Xiaoshuai Sun, Rongrong Ji*</sub></sup>  | [Paper](https://arxiv.org/abs/2312.12470)/[Code](https://github.com/Lsan2401/RMSIN)
|   06  | CVPR  | Theoretically Achieving Continuous Representation of Oriented Bounding Boxes  <br> <sup><sub>*Zikai Xiao, Guo-Ye Yang, Xue Yang, Tai-Jiang Mu, Junchi Yan, Shi-min Hu*</sub></sup>  | [Paper](https://arxiv.org/abs/2402.18975v1)/[Code](https://github.com/Jittor/JDet)
|   05  | CVPR  | Poly Kernel Inception Network for Remote Sensing Detection   <br> <sup><sub>*Xinhao Cai, Qiuxia Lai, Yuwei Wang, Wenguan Wang, Zeren Sun, Yazhou Yao*</sub></sup>  | [Paper](https://arxiv.org/abs/2403.06258)/Code
|   04  | AAAI | FRED: Towards a Full Rotation-Equivariance in Aerial Image Object Detection  <br> <sup><sub>*Chanho Lee, Jinsu Son, Hyounguk Shon, Yunho Jeon, Junmo Kim*</sub></sup>  | [Paper](https://arxiv.org/abs/2401.06159)/Code
|   03  | AAAI | Spatial Transform Decoupling for Oriented Object Detection   <br><sub><sup>*Hongtian Yu, Yunjie Tian, Qixiang Ye, Yunfan Liu*</sup></sub> | [Paper](https://arxiv.org/abs/2308.10561)/[Code](https://github.com/yuhongtian17/Spatial-Transform-Decoupling)
|   02  | IJCV | Oriented R-CNN and Beyond   <br><sub><sup>*Xingxing Xie, Gong Cheng, Jiabao Wang, Ke Li, Xiwen Yao & Junwei Han*</sup></sub> | [Paper](https://link.springer.com/article/10.1007/s11263-024-01989-w)/[Code](https://github.com/jbwang1997/OBBDetection)
|   01  | TGRS | ARS-DETR: Aspect Ratio-Sensitive Detection Transformer for Aerial Oriented Object Detection   <br><sub><sup>*Ying Zeng, Yushi Chen, Xue Yang, Qingyun Li, Junchi Yan*</sup></sub> | [Paper](https://arxiv.org/abs/2303.04989)/[Code](https://github.com/httle/ARS-DETR)





#### 2023

**No.** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :-
12 | arXiv | Adaptive Dense Pseudo Label Selection for Semi-supervised Oriented Object Detection   <br><sub><sup>*Tong Zhao, Qiang Fang, Shuohao Shi, Xin Xu*</sup></sub> | [Paper](https://arxiv.org/abs/2311.12608)/Code 
11 | ICCV  | Adaptive Rotated Convolution for Rotated Object Detection   <br><sub><sup>*Yifan Pu, Yiru Wang, Zhuofan Xia, Yizeng Han, Yulin Wang, Weihao Gan, Zidong Wang, Shiji Song, Gao Huang*</sup></sub> | [Paper](https://arxiv.org/abs/2303.07820)/[Code](https://github.com/LeapLabTHU/ARC) 
10 | ICCV  | Large Selective Kernel Network for Remote Sensing Object Detection  <br><sub><sup>*Yuxuan Li, Qibin Hou, Zhaohui Zheng, Ming-Ming Cheng, Jian Yang, and Xiang Li*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf)/[Code](https://github.com/zcablii/LSKNet)
09 | TIP   | Sampling Equivariant Self-attention Networks for Object Detection in Aerial Images <br><sub><sup>*Guo-Ye Yang; Xiang-Li Li; Zi-Kai Xiao; Tai-Jiang Mu; Ralph R. Martin; Shi-Min Hu*</sup></sub> | [Paper](https://arxiv.org/abs/2111.03420)/Code
08 | PR    | RoMP-Transformer: Rotational bounding box with Multi-level feature Pyramid Transformer for object detection   <br><sub><sup>*Joonhyeok Moon, Munsu Jeon, Siheon Jeong, Ki-Yong Oh*</sup></sub> | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323007641?dgcid=raven_sd_aip_email)/Code
07 | PAMI  | Detecting rotated objects as Gaussian distributions and its 3-D generalization  <br><sub><sup>*Yang, Xue and Zhang, Gefan and Yang, Xiaojiang and Zhou, Yue and Wang, Wentao and Tang, Jin and He, Tao and Yan, Junchi*</sup></sub> | [Paper](https://arxiv.org/abs/2209.10839)/Code 
06 | CVPR  | Dynamic Coarse-to-Fine Learning for Oriented Tiny Object Detection   <br><sub><sup>*Chang Xu, Jian Ding, Jinwang Wang, Chang Xu, Huai Yu, Lei Yu, Gui-Song Xia*</sup></sub> | Paper/[Code](https://github.com/Chasel-Tsui/mmrotate-dcfl)
05 | CVPR  | SOOD: Towards Semi-Supervised Oriented Object Detection  <br><sub><sup>*Wei Hua, Dingkang Liang, jingyu li, Xiaolong Liu, Zhikang Zou, Xiaoqing Ye, Xiang Bai*</sup></sub> | Paper/Code
04 | CVPR  | Knowledge Combination to Learn Rotated Detection Without Rotated Annotation <br><sub><sup>*Tianyu Zhu, Bryce Ferenczi, Pulak Purkait, Tom Drummond, Hamid Rezatofighi, Anton van den Hengel*</sup></sub> | [Paper](https://arxiv.org/abs/2304.02199)/Code
03 | CVPR  | Phase-Shifting Coder: Predicting Accurate Orientation in Oriented Object Detection <br><sub><sup>*Yi Yu, Feipeng Da*</sup></sub> | [Paper](https://arxiv.org/abs/2211.06368)/Code
02 | ICLR  | H2RBox: Horizontal Box Annotation is All You Need for Oriented Object Detection <br><sub><sup>*Yang, Xue and Zhang, Gefan and Li, Wentong and Wang, Xuehui and Zhou, Yue and Yan, Junchi*</sup></sub> | [Paper](https://arxiv.org/abs/2210.06742)/[Code](https://github.com/yangxue0827/h2rbox-mmrotate)
01 | TCSVT | AO2-DETR: Arbitrary-Oriented Object Detection Transformer <br><sub><sup>*Linhui Dai, Hong Liu, Hao Tang, Zhiwei Wu, Pinhao Song*</sup></sub> | [Paper](https://arxiv.org/abs/2205.12785)/[Code](https://github.com/Ixiaohuihuihui/AO2-DETR)





#### 2022

**No.** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :-
14 | TGRS | Anchor-free Oriented Proposal Generator for Object Detection <br><sub><sup>*Gong Cheng, Jiabao Wang, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2110.01931)/[Code](https://github.com/jbwang1997/AOPG)
13 | IJCV | On the Arbitrary-Oriented Object Detection: Classification Based Approaches Revisited <br><sub><sup>*Xue Yang, Junchi Yan*</sup></sub> | [Paper](https://link.springer.com/article/10.1007/s11263-022-01593-w?utm_source=toc&utm_medium=email&utm_campaign=toc_11263_130_5&utm_content=etoc_springer_20220428)/[Code](https://github.com/SJTU-Thinklab-Det/OHDet_Tensorflow)
12 | ECCV | EAutoDet: Efficient Architecture Search for Object Detection <br><sub><sup>*Xiaoxing Wang, Jiale Lin, Junchi Yan, Juanping Zhao, Xiaokang Yang*</sup></sub> | [Paper](https://arxiv.org/abs/2203.10747)/Code
11 | PAMI | SCRDet++: Detecting Small, Cluttered and Rotated Objects via Instance-Level Feature Denoising and Rotation Loss Smoothing  <br><sub><sup>*Xue Yang, Junchi Yan, et al.*</sup></sub> | [Paper](https://arxiv.org/pdf/2004.13316.pdf)/[Proj](https://yangxue0827.github.io/SCRDet++.html)
10 | TIP | GGHL: A General Gaussian Heatmap Label Assignment for Arbitrary-Oriented Object Detection  <br><sub><sup>*Zhanchao Huang, Wei Li, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2109.12848)/[Code](https://github.com/Shank2358/GGHL)
09 | TCSVT | RSDet++: Point-based Modulated Loss for More Accurate Rotated Object Detection <br><sub><sup>*Wen Qian, Xue Yang, et al.*</sup></sub> | [Paper](https://arxiv.org/pdf/2109.11906v1.pdf)/[Code](https://github.com/yangxue0827/RotationDetection)
08 | CVPR | Interactive Multi-Class Tiny-Object Detection  <br><sub><sup>*Chunggi Lee, Seonwook Park, Heon Song, Jeongun Ryu, Sanghoon Kim*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Interactive_Multi-Class_Tiny-Object_Detection_CVPR_2022_paper.pdf)/[Code](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection)
07 | CVPR | Weakly Supervised Rotation-Invariant Aerial Object Detection Network   <br><sub><sup>*Xiaoxu Feng, Gong Cheng, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Feng_Weakly_Supervised_Rotation-Invariant_Aerial_Object_Detection_Network_CVPR_2022_paper.pdf)/[Code](https://github.com/XiaoxFeng/RINet)
06 | CVPR | OSKDet: Towards Orientation-sensitive Keypoint Localization for Rotated Object Detection  <br><sub><sup>*Dongchen Lu, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lu_OSKDet_Orientation-Sensitive_Keypoint_Localization_for_Rotated_Object_Detection_CVPR_2022_paper.pdf)/Code
05 | CVPR | Oriented RepPoints for Aerial Object Detection   <br><sub><sup>*Wentong Li, Jianke Zhu*</sup></sub> | [Paper](https://arxiv.org/pdf/2105.11111.pdf)/[Code](https://github.com/LiWentomng/OrientedRepPoints)
04 | CVPR | ~~Canonical Voting: Towards Robust Oriented Bounding Box Detection in 3D Scenes~~   <br><sub><sup>*Yang You, Cewu Lu, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2011.12001)/[Code](https://github.com/qq456cvb/CanonicalVoting)
03 | AAAI | Shape-Adaptive Selection and Measurement for Oriented Object Detection  <br><sub><sup>*Liping Hou, Ke Lu, et al.*</sup></sub> | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-2171.HouL.pdf)/[Code](https://github.com/houliping/SASM)<br>[MMRotate](https://github.com/open-mmlab/mmrotate/tree/main/configs/sasm_reppoints)
02 | AAAI | Polygon-to-Polygon Distance Loss for Rotated Object Detection <br><sub><sup>*Yang Yang, Jifeng Chen, et al.*</sup></sub> | [Paper](https://www.aaai.org/AAAI22Papers/AAAI-8470.YangY.pdf)/Code/[Slides](https://aaai-2022.virtualchair.net/poster_aaai8470)
01 | TMM | AdaZoom: Towards Scale-Aware Large Scene Object Detection ``HBB``   <br><sub><sup>*Jingtao Xu, Yali Li, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9786052)/Code<br>[arXiv](https://arxiv.org/abs/2106.10409) |


#### 2021

**No.** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :- 
16 | **PAMI** | Gliding Vertex on the Horizontal Bounding Box for Multi-Oriented Object Detection  <br><sub><sup>*Yongchao Xu, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9001201?denied=)/[arXiv](https://arxiv.org/abs/1911.09358)<br>[Code](https://github.com/MingtaoFu/gliding_vertex) 
15 | TIP | <span style="white-space:nowrap;">GSDet: Object Detection in Aerial Images Based on Scale Reasoning&emsp;</span>  <br><sub><sup>*Wei Li, Wei Wei, Lei Zhang*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9411691)/Code
14 | **NeurIPS** | Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence  <br><sub><sup>*Xue Yang, Junchi Yan, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2106.01883)/[Code](https://github.com/yangxue0827/RotationDetection?utm_source=catalyzex.com)
13 | **ICCV** | Towards Rotation Invariance in Object Detection   <br><sub><sup>*Agastya Kalra, Guy Stoppi, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2109.13488v1)/[Code](https://github.com/akasha-imaging/ICCV2021)
12 | **ICCV** | Oriented R-CNN for Object Detection  <br><sub><sup>*X. Xie, Gong Cheng, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/2108.05699)/[Code](https://github.com/jbwang1997/OBBDetection)
11 | MM | Polar Ray: A Single-stage Angle-free Detector for Oriented Object Detection in Aerial Images  <br><sub><sup>*Shuai Liu, Huchuan Lu, et al.*</sup></sub> | [Paper](https://dl.acm.org/doi/pdf/10.1145/3474085.3475457)/Code
10 | **ICML** | Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss   <br><sub><sup>*Xue Yang,  Junchi Yan, et al.*</sup></sub> | [arXiv](https://arxiv.org/abs/2101.11952)/[Code](https://github.com/yangxue0827/RotationDetection)<br>[SUPP](http://proceedings.mlr.press/v139/yang21l/yang21l-supp.pdf)
09 | **CVPR** | ReDet: A Rotation-equivariant Detector for Aerial Object Detection  <br><sub><sup>*Jiaming Han, Jian Ding, Nan Xue, Gui-Song Xia*</sup></sub> |  [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Han_ReDet_A_Rotation-Equivariant_Detector_for_Aerial_Object_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/csuhan/ReDet) 
08 | **CVPR** | Beyond Bounding-Box: Convex-Hull Feature Adaptation for Oriented and Densely Packed Object Detection <br><sub><sup>*Z. Guo, Qixiang Ye, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Guo_Beyond_Bounding-Box_Convex-Hull_Feature_Adaptation_for_Oriented_and_Densely_Packed_CVPR_2021_paper.pdf)/[Code](https://github.com/SDL-GuoZonghao/BeyondBoundingBox)/<br>[Journal](https://ieeexplore.ieee.org/document/9668956)
07 | **CVPR** | Dense Label Encoding for Boundary Discontinuity Free Rotation Detection <br><sub><sup>*Xue Yang, Junchi Yan, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Yang_Dense_Label_Encoding_for_Boundary_Discontinuity_Free_Rotation_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/Thinklab-SJTU/DCL_RetinaNet_Tensorflow)
06 | CVPR | GAIA: A Transfer Learning System of Object Detection that Fits Your Needs `HBB`  <br><sub><sup>*X. Bu, Zhaoxiang Zhang, et al.*</sup></sub> | [Paper](https://arxiv.org/pdf/2106.11346.pdf)/[Code](https://github.com/GAIA-vision/GAIA-det)
05 | **AAAI** | Dynamic Anchor Learning for Arbitrary-Oriented Object Detection   <br><sub><sup>*Qi Ming, Zhiqiang Zhou, et al.*</sup></sub> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16336)/[Code](https://github.com/ming71/DAL)
04 | **AAAI** | Learning Modulated Loss for Rotated Object Detection `RSDet`   <br><sub><sup>*Wen Qian, Xue Yang, Junchi Yang, et al.*</sup></sub> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16347)/[arXiv](https://arxiv.org/abs/1911.08299)<br>[Code](https://github.com/Mrqianduoduo/RSDet-8P-4R) 
03 | **AAAI** | <span style="white-space:nowrap;">R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object &emsp;&emsp;</span>   <br><sub><sup>*Xue Yang, Junchi Yang, et al.*</sup></sub> | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16426)/[arXiv](https://arxiv.org/abs/1908.05612)<br>[Py](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection)/[TF](https://github.com/Thinklab-SJTU/R3Det_Tensorflow) 
02 | WACV | Oriented object detection in aerial images with box boundary-aware vectors   <br><sub><sup>*Yi, Jingru and Wu, Pengxiang and Liu, Bo and Huang, Qiaoying and Qu, Hui and Metaxas, Dimitris*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Yi_Oriented_Object_Detection_in_Aerial_Images_With_Box_Boundary-Aware_Vectors_WACV_2021_paper.pdf)/[Code](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection)
01 | PR | Gradient-Aligned Convolution Neural Network ``rotation invariance``   <br><sub><sup>*You Hao, Ping Hu, et al.*</sup></sub> | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005343?dgcid=raven_sd_aip_email)/Code




#### 2020

**No.** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :- 
10 | TIP | A Global-Local Self-Adaptive Network for Drone-View Object Detection <br><sub><sup>*Sutao Deng, et al.*</sup></sub> | Paper/Code
09 | TNNLS | CRPN-SFNet: A High-Performance Object Detector on Large-Scale Remote Sensing Images <br><sub><sup>*Qifeng Lin, et al.*</sup></sub> | Paper/Code
08 | JPRS | Orientation guided anchoring for geospatial object detection from remote sensing imagery <br><sub><sup>*Yongtao Yu, et al.*</sup></sub> | Paper/Code
07 | JPRS | <span style="white-space:nowrap;">Rotation-aware and multi-scale convolutional neural network for object detection in remote sensing images &emsp;</span>    <br><sub><sup>*Kun Fu, et al.*</sup></sub>  | Paper/Code
06 | JPRS | Oriented objects as pairs of middle lines <br><sub><sup>*Haoran Wei, et al.*</sup></sub>  | [Paper](https://arxiv.org/abs/1912.10694)/Code
05 | **ECCV** | Arbitrary-Oriented Object Detection with Circular Smooth Label `Conf`  <br> On the Arbitrary-Oriented Object Detection: Classification based Approaches Revisited `Journal`  <br><sub><sup>*Xue Yang, Junchi Yan, Tao He*</sup></sub>  | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58598-3_40)/[Code](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow)<br>[Code2](https://yangxue0827.github.io/CSL_GCL_OHDet.html)/[Journal](https://arxiv.org/abs/2003.05597)<br>[Proj](https://yangxue0827.github.io/CSL_GCL_OHDet.html)/[Data](https://yangxue0827.github.io/OHD-SJTU.html)|
04 | **ECCV** | PIoU Loss: Towards Accurate Oriented Object Detection in Complex Environments  <br><sub><sup>*Zhiming Chen, Weiyao Lin, et al.*</sup></sub>  | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58558-7_12)/[Code](https://github.com/clobotics/piou)
03 | ICME | Cascade Detector With Feature Fusion For Arbitrary-Oriented Objects In Remote Sensing Images   <br><sub><sup>*Liping Hou, Ke Lu, et al.*</sup></sub>  | Paper/Code
02 | **CVPR** | Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery <br><sub><sup>*Zhuo Zheng, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.html)/[Code](https://github.com/Z-Zheng/FarSeg)
01 | **CVPR** | <span style="white-space:nowrap;">Dynamic Refinement Network for Oriented and Densely Packed Object Detection&emsp;</span>  <br><sub><sup>*X. Pan, Changsheng Xu, et al.*</sup></sub>  | [Paper](https://arxiv.org/abs/2005.09973)/[Code](https://github.com/Anymake/DRN\_CVPR2020)



#### 2019

**No.** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :- 
09 | TIP | Learning Rotation-Invariant and Fisher Discriminative Convolutional Neural Networks for Object Detection <br><sub><sup>*Gong Cheng, Junwei Han, et al.*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/document/8445665)/[CVPR16](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cheng_RIFD-CNN_Rotation-Invariant_and_CVPR_2016_paper.pdf)<br>Code
08 | TGRS | CAD-Net: A Context-Aware Detection Network for Objects in Remote Sensing Imagery  <br><sub><sup>*Gongjie Zhang, Shijian Lu, Wei Zhang*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/document/8804364?denied=)/[arXiv](https://arxiv.org/pdf/1903.00857.pdf)<br>[Code](https://github.com/ZhangGongjie/CAD-Net)
07 | ICCV | Delving Into Robust Object Detection From Unmanned Aerial Vehicles: A Deep Nuisance Disentanglement Approach `HBB`  <br><sub><sup>*Zhenyu Wu, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wu_Delving_Into_Robust_Object_Detection_From_Unmanned_Aerial_Vehicles_A_ICCV_2019_paper.pdf)/[Code](https://github.com/VITA-Group/UAV-NDFT)
06 | **ICCV** | SCRDet: Towards More Robust Detection for Small, Cluttered and Rotated Objects <br><sub><sup>*Xue Yang, Junchi Yan, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yang_SCRDet_Towards_More_Robust_Detection_for_Small_Cluttered_and_Rotated_ICCV_2019_paper.pdf)/[Code](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow)
05 | ICCV | Clustered Object Detection in Aerial Images `HBB`   <br><sub><sup>*Fan Yang, Haibin Ling, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/1904.08008)/[Code](https://github.com/fyangneil/Clustered-Object-Detection-in-Aerial-Image)
04 | ICME | Cropping Region Proposal Network Based Framework for Efficient Object Detection on Large Scale Remote Sensing Images  <br><sub><sup>*Qifeng Lin, et al.*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/document/8784756?denied=)/Code
03 | **CVPR** | <span style="white-space:nowrap;">Learning RoI Transformer for Oriented Object Detection in Aerial Images&emsp;</span>  <br><sub><sup>*Jian Ding, Nan Xue, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.pdf)/[Code](https://github.com/dingjiansw101/RoITransformer_DOTA)
02 | CVPR | Precise Detection in Densely Packed Scenes `GOD`  <br><sub><sup>*Eran Goldman, Roei Herzig, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Goldman_Precise_Detection_in_Densely_Packed_Scenes_CVPR_2019_paper.pdf)/[Code](https://github.com/eg4000/SKU110K_CVPR19) 
01 | CVPR | Towards Universal Object Detection by Domain Attention   <br><sub><sup>*Xudong Wang, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Towards_Universal_Object_Detection_by_Domain_Attention_CVPR_2019_paper.pdf)/[Code](http://www.svcl.ucsd.edu/projects/universal-detection/)



#### 2018 

**No.** | **Pub.** | **Title**  | **Links**
:-: | :-: | :-  | :- 
03 | TIP | Random Access Memories: A New Paradigm for Target Detection in High Resolution Aerial Remote Sensing Images  <br><sub><sup>*Zhengxia Zou, Zhenwei Shi*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/document/8106808)/Code
02 | **CVPR** | DOTA: A Large-Scale Dataset for Object Detection in Aerial Images   <br><sub><sup>*Gui-Song Xia, Xiang Bai, Jian Ding, Zhen Zhu, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.pdf)/[Code](https://github.com/dingjiansw101/AerialDetection) 
01 | CVPR | <span style="white-space:nowrap;">ClusterNet: Detecting Small Objects in Large Scenes by Exploiting Spatio-Temporal Information `HBB`&emsp;</span>    <br><sub><sup>*R. Londe, Dong Zhang, Mubarak Shah*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/LaLonde_ClusterNet_Detecting_Small_CVPR_2018_paper.pdf)/[arXiv](https://arxiv.org/abs/1704.02694)<br>Code 




#### Before 2018

**No.** | **Year** | **Pub.** | **Title** | **Links**  
:-: |:-: | :-: | :-  | :- 
04 | 2016 | CVPR | RIFD-CNN: Rotation-Invariant and Fisher Discriminative Convolutional Neural Networks for Object Detection  <br><sub><sup>*Gong Cheng, Peicheng Zhou, Junwei Han*</sup></sub> | [Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Cheng_RIFD-CNN_Rotation-Invariant_and_CVPR_2016_paper.pdf)/Code|
03 | 2016 | TGRS | <span style="white-space:nowrap;">Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images&emsp;</span>   <br><sub><sup>*Gong Cheng, Peicheng Zhou, Junwei Han*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/abstract/document/7560644)/[Code1](https://pan.baidu.com/s/158JPjYQvXwnkrAqKF72Jgg#list/path=%2FRICNN)<br>[Code2](https://github.com/jiangruoqiao/RICNN_GongCheng_CVPR2015)|
02 | 2016 | ECCV | A Large Contextual Dataset for Classification, Detection and Counting of Cars with Deep Learning \[HBB\]  <br><sub><sup>*T. Nathan Mundhenk, et al.*</sup></sub>  | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_48)/Code
01 | 2015 | ICCV | Oriented Object Proposals   <br><sub><sup>*Shengfeng He, Rynson W. H. Lau*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Oriented_Object_Proposals_ICCV_2015_paper.pdf)/Code  |



## Instance Segmentation

**Year** | **Pub.** | **Title** | **Links**
:-: | :-: | :- | :- 
2024 | CVPR	| Rotated Multi-Scale Interaction Network for Referring Remote Sensing Image Segmentation   <br> <sup><sub>*Sihan Liu, Yiwei Ma, Xiaoqing Zhang, Haowei Wang, Jiayi Ji, Xiaoshuai Sun, Rongrong Ji*</sub></sup>  | [Paper](https://arxiv.org/abs/2312.12470)/[Code](https://github.com/Lsan2401/RMSIN)
2021 | TCYB | Semantic Attention and Scale Complementary Network for Instance Segmentation in Remote Sensing Images  <br> <sup><sub>*Tianyang Zhang, Licheng Jiao, et al.*</sub></sup> | [Paper](https://ieeexplore.ieee.org/document/9523594)/Code
2020 | CVPR | Foreground-Aware Relation Network for Geospatial Object Segmentation in High Spatial Resolution Remote Sensing Imagery   <br> <sup><sub>*Zhuo Zheng, et al.*</sub></sup> | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Foreground-Aware_Relation_Network_for_Geospatial_Object_Segmentation_in_High_Spatial_CVPR_2020_paper.html)/[Code](https://github.com/Z-Zheng/FarSeg)





## Small Object Detection 


**Year** | **Pub.** | **Title** | **Links**
:-: | :-: | :-  | :- 
2024 |  TCSVT | Save the Tiny, Save the All: Hierarchical Activation Network for Tiny Object Detection   <br> <sub><sup>*Guangqian Guo; Pengfei Chen; Xuehui Yu; Zhenjun Han; Qixiang Ye; Shan Gao*</sup></sub>  | [Paper](https://ieeexplore.ieee.org/document/10146336?source=tocalert&dld=Z21haWwuY29t)/Code
2023 | arXiv | Transformers in Small Object Detection: A Benchmark and Survey of State-of-the-Art    <br> <sub><sup>*Aref Miri Rekavandi, Shima Rashidi, Farid Boussaid, Stephen Hoefs, Emre Akbas, Mohammed Bennamoun*</sup></sub>  | [Paper](https://arxiv.org/abs/2309.04902)/[Code](https://github.com/arekavandi/Transformer-SOD)   
2023 |  ICCV | Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning   <br> <sub><sup>*Xiang Yuan, Gong Cheng, Kebing Yan, Qinghua Zeng, Junwei Han*</sup></sub>  | [Paper](https://arxiv.org/abs/2308.09534)/[Code](https://github.com/shaunyuan22/CFINet) 
2023 |  PAMI | Towards Large-Scale Small Object Detection: Survey and Benchmarks <br> <sub><sup>*Gong Cheng, Xiang Yuan, Xiwen Yao, Kebing Yan, Qinghua Zeng, Junwei Han*</sup></sub>  | [Paper](https://arxiv.org/abs/2207.14096)/[Data](https://shaunyuan22.github.io/SODA/)
2022 |  JPRS  | Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark <br> <sub><sup>*Chang Xu, Jinwang Wang, Wen Yang, Huai Yua, Lei Yua, Gui-SongXi*</sup></sub>  | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622001599#!)/[Code](https://github.com/jwwangchn/NWD)
2022 |  ECCV  | RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection <br> <sub><sup>*Chang Xu, Jinwang Wang, Wen Yang, Huai Yu, Lei Yu, Gui-Song Xia*</sup></sub>  | [Paper](https://arxiv.org/abs/2208.08738)/[Code](https://github.com/Chasel-Tsui/mmdet-rfla)
2022 | CVPR | QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection <br> <sub><sup>*Chenhongyi Yang, Zehao Huang, Naiyan Wang*</sup></sub> | [Paper](https://arxiv.org/abs/2103.09136)/[Code](https://github.com/ChenhongyiYang/QueryDet-PyTorch)
-- | -- | --  | -- | --
2021 | arXiv07 | A Survey on Deep Domain Adaptation and Tiny Object Detection Challenges, Techniques and Datasets <br> <sub><sup>*Muhammed Muzammul, Xi Li*</sup></sub> | [Paper](https://arxiv.org/abs/2107.07927)/Code
2021 | TMM | Extended Feature Pyramid Network for Small Object Detection <br> <sub><sup>*Chunfang Deng, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9409729)/Code
2021 | TMM | CrossNet: Detecting Objects as Crosses <br> <sub><sup>*Jiaxu Leng, Xinbo Gao, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9357941)/Code
2021 | CVPR | Dot distance for tiny object detection in aerial images  <br> <sub><sup>*Chang Xu, Jinwang Wang, Wen Yang, Lei Yu*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Xu_Dot_Distance_for_Tiny_Object_Detection_in_Aerial_Images_CVPRW_2021_paper.html)/Code
2021 | CVPR | Dogfight: Detecting Drones from Drones Videos <br> <sub><sup>*M. Ashraf, W. Sultani, Mubarak Shah*</sup></sub> | [Paper](https://arxiv.org/abs/2103.17242)/Code
2021 | WACV | Effective Fusion Factor in FPN for Tiny Object Detection <br> <sub><sup>*Yuqi Gong, Zhenjun Han, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Gong_Effective_Fusion_Factor_in_FPN_for_Tiny_Object_Detection_WACV_2021_paper.pdf)/Code
||---|---|---|---|
2020 | IJCV | Multi-task Generative Adversarial Network for Detecting Small Objects in the Wild <br> <sub><sup>*Yongqiang Zhang, Yancheng Bai, et al.*</sup></sub> | [Paper](https://link.springer.com/article/10.1007/s11263-020-01301-6)/[CVPR18](https://openaccess.thecvf.com/content_cvpr_2018/papers/Bai_Finding_Tiny_Faces_CVPR_2018_paper.pdf)<br>[ECCV18](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yongqiang_Zhang_SOD-MTGAN_Small_Object_ECCV_2018_paper.pdf)/Code 
2020 | TCYB | Context-Aware Block Net for Small Object Detection <br> <sub><sup>*Lisha Cui, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/9151360)/Code
2020 | MM | CODAN: Counting-driven Attention Network for Vehicle Detection in Congested Scenes <br> <sub><sup>*Wei Li, et al.*</sup></sub> | [Paper](https://dl.acm.org/doi/10.1145/3394171.3413945)/Code|
||---|---|---|---|
2019 | arXiv02 | Augmentation for small object detection <br> <sub><sup>*Mate Kisantal, et al.*</sup></sub> | [Paper](https://arxiv.org/abs/1902.07296)/Code
2019 | TCSVT | Detecting Small Objects Using a Channel-Aware Deconvolutional Network   <br> <sub><sup>*Kaiwen Duan, Dawei Du, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8669953)/Code
2019 | TGRS | R2 -CNN: Fast Tiny Object Detection in Large-Scale Remote Sensing Images <br> <sub><sup>*J. Pang, Jianping Shi, et al.*</sup></sub> | [Paper](https://ieeexplore.ieee.org/document/8672899)/Code
2019 | ICCV | Better to Follow, Follow to Be Better: Towards Precise Supervision of Feature Super-Resolution for Small Object Detection <br> <sub><sup>*Junhyug Noh, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Noh_Better_to_Follow_Follow_to_Be_Better_Towards_Precise_Supervision_ICCV_2019_paper.html)/Code
2019 | ICCV | Miss Detection vs. False Alarm: Adversarial Learning for Small Object Segmentation in Infrared Images <br> <sub><sup>*Huan Wang, Luping Zhou, Lei Wang*</sup></sub> | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Miss_Detection_vs._False_Alarm_Adversarial_Learning_for_Small_Object_ICCV_2019_paper.pdf)/[Code](https://github.com/wanghuanphd/MDvsFA_cGAN)
2019 | MM | Small and Dense Commodity Object Detection with Multi-Receptive Field Attention <br> <sub><sup>*Zhong  Ji, Yanwei Pang, et al.*</sup></sub> | [Paper](https://dl.acm.org/doi/abs/10.1145/3343031.3351064)/Code 
||---|---|---|---|
2018 | CVPR | Finding Tiny Faces in the Wild With Generative Adversarial Network <br> <sub><sup>*Yancheng Bai, Bernard Ghanem, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Bai_Finding_Tiny_Faces_CVPR_2018_paper.html)/Code
2017 | ICCV | Focal Loss for Dense Object Detection <br> <sub><sup>*Tsung-Yi, Kaiming He, et al.*</sup></sub> | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)/[Code](https://github.com/facebookresearch/Detectron)|
2017 | CVPR | Perceptual Generative Adversarial Networks for Small Object Detection  <br> <sub><sup>*Jianan Li, Tingfa Xu, et al.*</sup></sub>  | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Perceptual_Generative_Adversarial_CVPR_2017_paper.pdf)/Code





## UAV Object Detection

**Year** | **Pub.** | **Title** | **Authors** | **Links**
:-: | :-: | :-  | :- | :- 
2021 | CVPR | Detection, Tracking, and Counting Meets Drones in Crowds: A Benchmark | Longyin Wen, Dawei Du, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wen_Detection_Tracking_and_Counting_Meets_Drones_in_Crowds_A_Benchmark_CVPR_2021_paper.pdf)/[Code](https://github.com/VisDrone/DroneCrowd)|
||---|---|---|---|
2020 | arXiv | Vision Meets Drones: Past, Present and Future | Pengfei Zhu, Dawei Du, et al. | [Paper](https://arxiv.org/pdf/2001.06303.pdf)/[Code](https://github.com/VisDrone/VisDrone-Dataset)|
2020 | IJCV | The Unmanned Aerial Vehicle Benchmark: Object Detection, Tracking and Baseline | Hongyang Yu, Dawei Du, et al. | [Paper](https://link.springer.com/article/10.1007/s11263-019-01266-1#Fn2)/[Proj](https://sites.google.com/view/daweidu/projects/uavdt)<br>[ECCV18](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf)|
2020 | MM | MOR-UAV: A Benchmark Dataset and Baselines for Moving Object Recognition in UAV Videos | M. Mandal, Lav Kumar, S. Vipparthi | [Paper](https://arxiv.org/pdf/2008.01699.pdf)/[Code](https://github.com/murari023/mor-uav)|
2020 | MM | <span style="white-space:nowrap;">Guided Attention Network for Object Detection and Counting on Drones&emsp;</span> | <span style="white-space:nowrap;">Y. Cai, Dawei Du, et al.&emsp;</span> | [Paper](https://arxiv.org/pdf/1909.11307.pdf)/[Code](https://isrc.iscas.ac.cn/gitlab/research/ganet/-/tree/master)|









## Dataset

**Year** | **Name** | **Paper** | **Pub.** 
:-: | :-: | :-  | :-
2022 | [SODA](https://shaunyuan22.github.io/SODA/) | <font size=2>Towards Large-Scale Small Object Detection: Survey and Benchmarks `[OBB]`</font> | [Paper](https://arxiv.org/abs/2207.14096) 
2021 | [**FAIR1M**](http://gaofen-challenge.com) | <font size=2>FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery `[OBB]`</font> | [arXiv](https://arxiv.org/abs/2103.05569)<br>[Intro-ch](https://mp.weixin.qq.com/s/r5aZf_TLoJ3BGb_ong95Vw)
2021 | [SaRNet](https://github.com/michaelthoreau/SearchAndRescueNet) | <font size=2>SaRNet: A Dataset for Deep Learning Assisted Search and Rescue with Satellite Imagery `[HBB]`</font> | [arXiv](https://arxiv.org/abs/2107.12469)
2021 | [AI-TOD](https://github.com/jwwangchn/AI-TOD) | <font size=2>Tiny Object Detection in Aerial Images `[HBB]`</font> | [ICPR](https://drive.google.com/file/d/1IiTp7gilwDCGr8QR_H9Covz8aVK7LXiI/view) 
2019 | [DIOR](https://gcheng-nwpu.github.io/#Datasets) | <font size=2>Object detection in optical remote sensing images: A survey and a new benchmark `[HBB]`</font> | [ISPRSJ](https://www.sciencedirect.com/science/article/abs/pii/S0924271619302825)
2019 | [iSAID](https://captain-whu.github.io/iSAID/) | <font size=2>iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images</font> | [CVPRW](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Zamir_iSAID_A_Large-scale_Dataset_for_Instance_Segmentation_in_Aerial_Images_CVPRW_2019_paper.pdf)
2019 | [HRRSD](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset) | <font size=2>Hierarchical and Robust Convolutional Neural Network for Very High-Resolution Remote Sensing Object Detection</font> | [TGRS](https://ieeexplore.ieee.org/document/8676107)
2019 | [VRAI](https://github.com/muzishen/awesome-vehicle_reid-dataset) | <font size=2>Vehicle Re-identification in Aerial Imagery: Dataset and Approach</font> | [ICCV](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Vehicle_Re-Identification_in_Aerial_Imagery_Dataset_and_Approach_ICCV_2019_paper.pdf) 
2019 | [ITCVD](https://research.utwente.nl/en/datasets/itcvd-dataset) | <font size=2>Vehicle Detection in Aerial Images</font> | [PERS](https://www.ingentaconnect.com/content/asprs/pers/2019/00000085/00000004/art00014#)
2019 | [Aerial<br>Elephant](https://zenodo.org/record/3234780#.YUsUnGYzahw) | <font size=2>The Aerial Elephant Dataset: A New Public Benchmark for Aerial Object Detection</font> | [CVPRW](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Naude_The_Aerial_Elephant_Dataset_A_New_Public_Benchmark_for_Aerial_CVPRW_2019_paper.pdf)
2018 | [**DOTA**](https://captain-whu.github.io/DOTA/) | <font size=2>DOTA: A Large-Scale Dataset for Object Detection in Aerial Images</font> | [CVPR](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xia_DOTA_A_Large-Scale_CVPR_2018_paper.pdf)/[Kit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
2018 | [xView](http://xviewdataset.org/) | <font size=2>xView: Objects in Context in Overhead Imagery `[HBB]`</font> | [arXiv](https://arxiv.org/abs/1802.07856)/[Kit](https://github.com/dellemc-hpc-ai/satellite_imagery_demo) 
2018 | [VisDrone](http://aiskyeye.com/) | <font size=2>Vision Meets Drones: Past, Present and Future  `[HBB]`</font> | [arXiv](https://arxiv.org/pdf/2001.06303.pdf)/[Data](https://github.com/VisDrone/VisDrone-Dataset)
2018 | [LPODC](https://github.com/xyzxinyizhang/2018-DAC-System-Design-Contest) | <font size=2>DAC-SDC Low Power Object Detection Challenge for UAV Applications `[HBB]`</font> | [PAMI](https://ieeexplore.ieee.org/document/8787881)
2018 | [UAVDT](https://sites.google.com/view/daweidu/projects/uavdt?authuser=0) | <font size=2>The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking</font> | [ECCV](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf)  
2016 | [NWPU<br>VHR-10](https://gcheng-nwpu.github.io/#Datasets) | <font size=2>Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images</font> | [TGRS](https://ieeexplore.ieee.org/document/7560644)
2016 | [RSOD](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-) | <font size=2>Accurate Object Localization in Remote Sensing Images Based on Convolutional Neural Networks</font> | [TGRS](https://ieeexplore.ieee.org/abstract/document/7827088) 
2016 | [**HRSC<br>2016**](https://sites.google.com/site/hrsc2016/) | <font size=2>A High Resolution Optical Satellite Image Dataset for Ship Recognition and Some New Baselines</font> | [ICPRAM](https://www.scitepress.org/Papers/2017/61206/61206.pdf)<br>[Kaggle](https://www.kaggle.com/guofeng/hrsc2016)
2015 | [VEDAI](https://downloads.greyc.fr/vedai/) | <font size=2>Vehicle Detection in Aerial Imagery: A small target detection benchmark</font> | [JVCIR](https://hal.archives-ouvertes.fr/hal-01122605v2/document)
2014 | [**UCAS-AOD**](https://hyper.ai/datasets/5419) | <font size=2>Orientation robust object detection in aerial images using deep convolutional neural network</font> | [ICIP](https://ieeexplore.ieee.org/document/7351502)




## Appendix

### GOD


**Year** | **Pub.** | **Title** | **Authors** | **Links**
:-: | :-: | :-  | :- | :-
2021 | arXiv09 | Progressive Hard-case Mining across Pyramid Levels in Object Detection | Binghong Wu, et al. | [Paper](https://arxiv.org/abs/2109.07217)/[Code](https://github.com/zimoqingfeng/UMOP) |
2021 | arXiv04 | Slender Object Detection: Diagnoses and Improvements | Zhaoyi Wan, Yimin Chen, et al. | [Paper](https://arxiv.org/abs/2011.08529)/[Code](https://github.com/wanzysky/SlenderObjDet) |
2021 | arXiv01 | Focal and Efficient IOU Loss for Accurate Bounding Box Regression | Yi-Fan Zhang, Liang Wang, et al. | [Paper](https://arxiv.org/abs/2101.08158)/Code |
2021 | NeurIPS | You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection | Yuxin Fang, Xinggang Wang, et al. | [Paper](https://arxiv.org/pdf/2106.00666.pdf)/[Code](https://github.com/hustvl/YOLOS) |
2021 | PAMI | Scale Normalized Image Pyramids with AutoFocus for Object Detection | Bharat Singh, Mahyar Najibi, et al. | [Paper](https://arxiv.org/abs/2102.05646)/[Code](https://github.com/mahyarnajibi/SNIPER) |
2021 | IJCV | Compositional Convolutional Neural Networks: A Robust and Interpretable Model for Object Recognition Under Occlusion | Adam Kortylewski, et al. | [Paper](https://link.springer.com/article/10.1007/s11263-020-01401-3)/Code|
2021 | IJCV | Scale-Aware Domain Adaptive Faster R-CNN | Y. Chen, D. Dai, Luc Van Gool, et al. | [Paper](https://link.springer.com/article/10.1007/s11263-021-01447-x#Abs1)/[Code](https://github.com/yuhuayc/sa-da-faster)
2021 | IJCV | Guided Attention in CNNs for Occluded Pedestrian Detection and Re-identification | S. Zhang, Di Chen, Jian Yang, Bernt Schiele | [Paper](https://link.springer.com/article/10.1007/s11263-021-01461-z)/Code|
2021 | ICCV | SOTR: Segmenting Objects With Transformers | Ruohao Guo, Dantong Niu, Liao Qu, Zhenbo Li | Paper/Code|
2021 | ICCV | Dynamic DETR: End-to-End Object Detection With Dynamic Attention | Xiyang Dai, Lu Yuan; Lei Zhang, et al. | Paper/Code|
2021 | ICCV | CrossDet: Crossline Representation for Object Detection | Heqian Qiu, Hongliang Li, et al. | Paper/Code |
2021 | CVPR | UP-DETR: Unsupervised Pre-training for Object Detection with Transformers | Z. Dai, J. Chen, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_UP-DETR_Unsupervised_Pre-Training_for_Object_Detection_With_Transformers_CVPR_2021_paper.pdf)/[Code](https://github.com/dddzg/up-detr)|
2021 | CVPR | Adaptive Image Transformer for One-Shot Object Detection | Ding-Jie Chen, He-Yen Hsieh, Tyng-Luh Liu | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Adaptive_Image_Transformer_for_One-Shot_Object_Detection_CVPR_2021_paper.pdf)/Code|
2021 | CVPR | Scale-aware Automatic Augmentation for Object Detection | Yukang Chen, Jiaya Jia, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Scale-Aware_Automatic_Augmentation_for_Object_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/dvlab-research/SA-AutoAug) |
2021 | CVPR | Sparse R-CNN: End-to-End Object Detection with Learnable Proposals | Peize Sun,  Ping Luo, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Sun_Sparse_R-CNN_End-to-End_Object_Detection_With_Learnable_Proposals_CVPR_2021_paper.pdf)/[Code](https://github.com/PeizeSun/SparseR-CNN)<br>[arXiv](https://arxiv.org/abs/2011.12450) |
2021 | ICLR | Deformable DETR: Deformable Transformers for End-to-End Object Detection | Xizhou Zhu, Xiaogang Wang, Jifeng Dai, et al. | [Paper](https://openreview.net/pdf?id=gZ9hCDWe6ke)/[Code](https://github.com/fundamentalvision/Deformable-DETR)|
2021 | ICLR | On the Universality of Rotation Equivariant Point Cloud Networks |Nadav Dym, Haggai Maron | [Paper](https://openreview.net/pdf?id=6NFBvWlRXaG)/Code|
2021 | AAAI | Rethinking Object Detection in Retail Stores | Yuanqiang Cai, Dawei Du, etc | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16178)/[Code](https://isrc.iscas.ac.cn/gitlab/research/locount-dataset) |
||---|---|---|---|
2020 | NeurIPS | RepPoints V2: Verification Meets Regression for Object Detection | Yihong Chen, Zheng Zhang, et al. | [Paper](https://arxiv.org/abs/2007.08508)/[Code](https://github.com/Scalsol/RepPointsV2)
2020 | TIP | Self-Supervised Feature Augmentation for Large Image Object Detection | Xingjia Pan, et al. | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9094016&casa_token=3pTpBlwlFFcAAAAA:fj8ElKNHVM2HDdizamiYUPg52LwC95dnlBcI2qK-bLAA1Sf4QyDVz2D6495NxEwW2tcm2Bw&tag=1)/Code|
2020 | ECCV | End-to-End Object Detection with Transformers `DETR` | Carion N, et al. | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf)/[Code](https://github.com/facebookresearch/detr)
2020 | ECCV | Learning Data Augmentation Strategies for Object Detection | Barret Zoph, Quoc V. Le, et al. | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720562.pdf)/[Code](https://github.com/tensorflow/tpu/tree/master/models/official/detection)|
2020 | ECCV | Learning to Separate: Detecting Heavily-Occluded Objects in Urban Scenes | Chenhongyi Yang, et al. | [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630511.pdf)/[Code](https://github.com/ChenhongyiYang/SG-NMS)|
||---|---|---|---|
2019 | IJCV | CornerNet: Detecting Objects as Paired Keypoints | Hei Law, Jia Deng | [Paper](https://link.springer.com/article/10.1007/s11263-019-01204-1#Abs1)/[arXiv](https://arxiv.org/abs/1808.01244)<br>[ECCV18](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hei_Law_CornerNet_Detecting_Objects_ECCV_2018_paper.pdf)/[Code](https://github.com/princeton-vl/CornerNet)|
2019 | IJCV | Corner Detection Using Multi-directional Structure Tensor with Multiple Scales | Weichuan Zhang, Changming Sun | [Paper](https://link.springer.com/article/10.1007/s11263-019-01257-2#Abs1)/Code|
2019 | IJCV | Hierarchical Attention for Part-Aware Face Detection | Shuzhe Wu, Meina Kan, Shiguang Shan, Xilin Chen | Paper/Code|
2019 | TIP | Combining Faster R-CNN and Model-Driven Clustering for Elongated Object Detection | Fen Fang, et al. | Paper/Code|
2019 | arXiv | Deep Learning for 2D and 3D Rotatable Data: An Overview of Methods | Luca Della Libera, Daniel Cremers, et al. | [Paper](https://arxiv.org/abs/1910.14594)/Code
2019 | ICCV | AutoFocus: Efficient Multi-Scale Inference | Mahyar Najibi, Bharat Singh, Larry S. Davis | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Najibi_AutoFocus_Efficient_Multi-Scale_Inference_ICCV_2019_paper.pdf)/[Code](https://github.com/mahyarnajibi/SNIPER)
2019 | ICCV | Scale-Aware Trident Networks for Object Detection | Yanghao Li, Naiyan Wang, et al. | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Li_Scale-Aware_Trident_Networks_for_Object_Detection_ICCV_2019_paper.html)/[Code](https://github.com/TuSimple/simpledet)
2019 | ICCV | RepPoints: Point Set Representation for Object Detection | Ze Yang, Han Hu, et al. | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_RepPoints_Point_Set_Representation_for_Object_Detection_ICCV_2019_paper.html)/[Code](https://github.com/microsoft/RepPoints)|
2019 | CVPR | Self-Supervised Representation Learning by Rotation Feature Decoupling | Zeyu Feng, Chang Xu, Dacheng Tao | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.html)/[Code](https://github.com/philiptheother/FeatureDecoupling)
2019 | ICLR | A rotation-equivariant convolutional neural network model of primary visual cortex | Alexander S. Ecker, et al. | [Paper](https://openreview.net/pdf?id=H1fU8iAqKX)/[Code](https://github.com/aecker/cnn-sys-ident)|
2019 | ICLR | RotDCF: Decomposition of Convolutional Filters for Rotation-Equivariant Deep Networks | Xiuyuan Cheng, et al. | [Paper](https://openreview.net/pdf?id=H1gTEj09FX)/[Code](https://github.com/ZichenMiao/RotDCF)|
||---|---|---|---|
2018 | PAMI | Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks | Luc Van Gool, et al. | Paper/Code|
2018 | TIP | Joint Hand Detection and Rotation Estimation Using CNN | Xiaoming Deng, et al. | Paper/Code|
2018 | ECCV | Occlusion-aware R-CNN: Detecting Pedestrians in a Crowd | Shifeng Zhang, et al. | [Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper.pdf)/Code|
2018 | ECCV | SAN: Learning Relationship between Convolutional Features for Multi-Scale Object Detection | Yonghyun Kim, et al. | [Paper](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Kim_SAN_Learning_Relationship_ECCV_2018_paper.pdf)/Code|
2018 | NeurIPS | SNIPER: Efficient Multi-Scale Training | Bharat Singh, Mahyar Najibi, Larry S. Davis | [Paper](https://arxiv.org/abs/1805.09300)/[Code](https://github.com/mahyarnajibi/SNIPER)
2018 | CVPR | An Analysis of Scale Invariance in Object Detection - SNIP | Bharat Singh, Larry S. Davis | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html)/[Code](https://github.com/bharatsingh430/snip)
2018 | ICLR | <span style="white-space:nowrap;">Unsupervised Representation Learning by Predicting Image Rotations&emsp;</span>  | <span style="white-space:nowrap;">S. Gidaris, P. Singh, et al.&emsp;</span> | [Paper](https://openreview.net/pdf?id=S1v4N2l0-)/[Code](https://github.com/gidariss/FeatureLearningRotNet) |



### Arbitrarily-Oriented Text Detection

**Year** | **Pub.** | **Title** | **Authors** | **Links**
:-: | :-: | :-  | :- | :-
2021 | ICCV | Adaptive Boundary Proposal Network for Arbitrary Shape Text Detection | Shi-Xue Zhang, et al. | [Paper](https://arxiv.org/abs/2107.12664)/[Code](https://github.com/GXYM/TextBPN) 
2021 | PAMI | PAN++: Towards Efficient and Accurate End-to-End Spotting of Arbitrarily-Shaped Text | Wenhai Wang, et al. | Paper/Code
2021 | PAMI | Towards End-to-End Text Spotting in Natural Scenes | Peng Wang, Hui Li, Chunhua Shen | Paper/Code
2021 | PAMI | Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes | Minghui Liao, Pengyuan Lyu, et al. | [Paper](https://ieeexplore.ieee.org/document/8812908)/[ECCV18]()
2021 | IJCV | Exploring the Capacity of an Orderless Box Discretization Network for Multi-orientation Scene Text Detection | Yuliang Liu, Chunhua Shen, et al. | [Paper](https://link.springer.com/article/10.1007/s11263-021-01459-7)/[Code](https://github.com/Yuliang-Liu/Box_Discretization_Network)
2021 | TIP | Arbitrarily Shaped Scene Text Detection With a Mask Tightness Text Detector | Yuliang Liu, Lianwen Jin, Chuanming Fang | Paper/Code
2021 | TIP | SLOAN: Scale-Adaptive Orientation Attention Network for Scene Text Recognition | Pengwen Dai, Hua Zhang, Xiaochun Cao | Paper/Code
2021 | MM | Mask is All You Need: Rethinking Mask R-CNN for Dense and Arbitrary-Shaped Scene Text Detection | Xugong Qin, Yu Zhou, et al. | Paper/Code
2021 | CVPR | Fourier Contour Embedding for Arbitrary-Shaped Text Detection | Yiqin Zhu, Lianwen Jin, et al. | [Paper](https://arxiv.org/abs/2104.10442)/Code
2021 | CVPR | Progressive Contour Regression for Arbitrary-Shape Scene Text Detection | Pengwen Dai, et al. | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_Progressive_Contour_Regression_for_Arbitrary-Shape_Scene_Text_Detection_CVPR_2021_paper.pdf)/[Code](https://github.com/dpengwen/PCR)
2021 | CVPR | MOST: A Multi-Oriented Scene Text Detector with Localization Refinement | Minghang He, Xiang Bai, et al. | [Paper](https://arxiv.org/abs/2104.01070)/Code
2021 | AAAI | MANGO: A Mask Attention Guided One-Stage Scene Text Spotter |Liang Qiao, Ying Chen, et al. | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16348)/Code
2021 | AAAI | PGNet: Real-time Arbitrarily-Shaped Text Spotting with Point Gathering Network | Pengfei Wang, et al.  | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16383)/Code
||---|---|---|---|
2020 | TIP | ASTS: A Unified Framework for Arbitrary Shape Text Spotting | Juhua Liu, Zhe Chen, Bo Du, Dacheng Tao | Paper/Code
2020 | TIP | Text Co-Detection in Multi-View Scene | Chuan Wang, Huazhu Fu, et al. | Paper/Code
2020 | ECCV | Character Region Attention For Text Spotting | Youngmin Baek, et al. | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_30)/Code 
2020 | CVPR | ContourNet: Taking a Further Step Toward Accurate Arbitrary-Shaped Scene Text Detection | Yuxin Wang, Zilong Fu, et al. | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ContourNet_Taking_a_Further_Step_Toward_Accurate_Arbitrary-Shaped_Scene_Text_CVPR_2020_paper.html)/[Code](https://github.com/wangyuxin87/ContourNet)
2020 | CVPR | Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection | Shi-Xue Zhang, et al. | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Deep_Relational_Reasoning_Graph_Network_for_Arbitrary_Shape_Text_Detection_CVPR_2020_paper.pdf)/[Code](https://github.com/GXYM/DRRG)
2020 | AAAI | Text Perceptron: Towards End-to-End Arbitrary-Shaped Text Spotting | Liang Qiao, et al. | [Paper](https://aaai.org/ojs/index.php/AAAI/article/view/6864)/Code
2020 | AAAI | All You Need Is Boundary: Toward Arbitrary-Shaped Text Spotting | Hao Wang, Xiang Bai, et al. | [Paper](https://aaai.org/ojs/index.php/AAAI/article/view/6896)/Code
||---|---|---|---|
2019 | ICCV | Efficient and Accurate Arbitrary-Shaped Text Detection With Pixel Aggregation Network | Wenhai Wang, et al. | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Efficient_and_Accurate_Arbitrary-Shaped_Text_Detection_With_Pixel_Aggregation_Network_ICCV_2019_paper.html)/[Code](https://github.com/WenmuZhou/PAN.pytorch)
2019 | ICCV | TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting | Wei Feng, Cheng-Lin Liu, et al. |[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf)/Code
2019 | CVPR | Learning Shape-Aware Embedding for Scene Text Detection | Zhuotao Tian, Jiaya Jia, et al. |[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tian_Learning_Shape-Aware_Embedding_for_Scene_Text_Detection_CVPR_2019_paper.pdf)/Code
2019 | CVPR | Arbitrary Shape Scene Text Detection with Adaptive Text Region Representation | Xiaobing Wang, Cheng-Lin Liu, et al. |[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Arbitrary_Shape_Scene_Text_Detection_With_Adaptive_Text_Region_Representation_CVPR_2019_paper.pdf)/Code
2019 | CVPR | Towards Robust Curve Text Detection with Conditional Spatial Expansion |Zichuan Liu, Guosheng Lin, et al.|[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Towards_Robust_Curve_Text_Detection_With_Conditional_Spatial_Expansion_CVPR_2019_paper.pdf)/Code
2019 | CVPR | Shape Robust Text Detection with Progressive Scale Expansion Network | Wenhai Wang, , Tong Lu, et al. |[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.pdf)/[Code](https://github.com/whai362/PSENet)
2019 | CVPR | Look More Than Once: An Accurate Detector for Text of Arbitrary Shapes | Chengquan Zhan, et al. |[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Look_More_Than_Once_An_Accurate_Detector_for_Text_of_CVPR_2019_paper.pdf)/Code
||---|---|---|---|
2018 | TIP | Multi-Oriented and Multi-Lingual Scene Text Detection With Direct Regression | Wenhao He, Cheng-Lin Liu, et al. | 
2018 | TIP | TextBoxes++: A Single-Shot Oriented Scene Text Detector | Minghui Liao, Baoguang She, Xiang Bai | 
2018 | TMM | Arbitrary-Oriented Scene Text Detection via Rotation Proposals | Jianqi  Ma, et al. | Paper/Code
2018 | ECCV | TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes | Shangbang Long, Jiaqiang Ruan, et al. | 
2018 | CVPR | Geometry-Aware Scene Text Detection with Instance Transformation Network|Fangfang Wang, et al.|[Paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1653.pdf)/[Code](https://github.com/LianaWang/itn)
2018 | CVPR | Rotation-Sensitive Regression for Oriented Scene Text Detection | Minghui Liao, Xiang Bai, et al.  | 
2018 | CVPR | AON: Towards Arbitrarily-Oriented Text Recognition||
2018 | CVPR | FOTS: Fast Oriented Text Spotting With a Unified Network||
2018 | CVPR | Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation||
||---|---|---|---|
2017 | TIP | Tracking Based Multi-Orientation Scene Text Detection: A Unified Framework With Dynamic Programming | Chun Yang, Junchi Yan, et al. | 
2017 | ICCV | Deep Direct Regression for Multi-Oriented Scene Text Detection||
2017 | ICCV | Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework | <span style="white-space:nowrap;">M. Busta, L. Neumann, Jiri Matas&emsp;</span> | [Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Busta_Deep_TextSpotter_An_ICCV_2017_paper.html)/Code
2017 | CVPR | <span style="white-space:nowrap;">Deep Matching Prior Network: Toward Tighter Multi-oriented Text Detection&emsp;</span> | Yuliang Liu, Lianwen Jin | [Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Deep_Matching_Prior_CVPR_2017_paper.html)/Code



### Related Resources

- [Awesome Tiny Object Detection](https://github.com/knhngchn/awesome-tiny-object-detection)
- [Awesome Semantic Understanding for Aerial Scene](https://github.com/tzxiang/awesome-semantic-understanding-for-aerial-scene)


