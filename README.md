# UAV-DETR
# ​**UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery**

This is the official implementation of the paper:
- ​**[UAV-DETR: Efficient End-to-End Object Detection for Unmanned Aerial Vehicle Imagery](https://arxiv.org/abs/2501.01855)**

## 🚀 Updates
- ​**[2024.10]**​ Release UAV-DETR-R50, UAV-DETR-R18.
- ​**[2025.01]**​ The initial version of the paper has been uploaded to arXiv.
- ​**[2025.03]**​ Release UAV-DETR-EV2. Fixed some bugs.

- 🔥 ​**UAV-DETR**

---

## Experimental Results on the VisDrone-2019-DET Dataset

| ​**Model**​            | ​**Backbone**​         | ​**Input Size**​ | ​**Params (M)**​ | ​**GFLOPs**​ | ​**AP**​  | ​**AP$_{50}$**​ |
|----------------------|---------------------|----------------|----------------|------------|---------|---------------|
| UAV-DETR-R50 (Ours)  | EfficientFormerV2   | 640×640        | 12.1           | 33.3       | 28.2    | 46.7          |
| UAV-DETR-R18 (Ours)  | ResNet18            | 640×640        | 20.5           | 64.3       | ​**29.8**| ​**48.8**​      |
| UAV-DETR-R50 (Ours)  | ResNet50            | 640×640        | 44.4           | 161.4      | ​**31.5**| ​**51.1**​      |

---

## Experimental Results on UAVVaste Dataset

| ​**Model**​             | ​**Params (M)**​ | ​**GFLOPs**​ | ​**AP**​  | ​**AP$_{50}$**​ |
|-----------------------|----------------|------------|---------|---------------|
| UAV-DETR-R50 (Ours)   | 44.4           | 161.4      | 37.5    | 75.9          |
| UAV-DETR-R18 (Ours)   | 20.5           | 64.3       | 35.1    | 72.1          |
| UAV-DETR-EV2 (Ours)   | 12.1           | 33.3       | 33.7    | 70.6          |

---

## Ablation Study

| ​**Model Configuration**​ | ​**AP**​  | ​**AP$_{50}$**​ |
|-------------------------|---------|---------------|
| Baseline                | 26.7    | 44.6          |
| Baseline + Inner-SIoU   | 27.1    | 45.3          |
| Baseline + MSFF-FE      | 28.4    | 46.9          |
| Baseline + MSFF-FE + FD | 28.4    | 47.1          |
| ​**Full Model**​          | ​**29.8**​ | ​**48.8**​      |

---

## 📍 Environment
- torch 1.13.1+cu11.7 
- torchvision 0.14.1+cuda11.7 
- Ubuntu 20.04

---
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ValiantDiligent/UAV-DETR&type=Date)](https://www.star-history.com/#ValiantDiligent/UAV-DETR&Date)

如果仍有疑问，请邮件联系：zhanghx23@m.fudan.edu.cn
