# Benchmark Evaluation of DenseNet121-Res224-All

## Table 1. Cross-Dataset Benchmark Performance of DenseNet121-Res224-All

| Dataset | Pathologies Evaluated | Mean AUC | Detailed AUC Scores |
|---|---|---:|---|
| **NIH ChestX-ray14** | 14 | **0.77** | Atelectasis (0.76), Cardiomegaly (0.88), Consolidation (0.77), Edema (0.85), Effusion (0.85), Emphysema (0.73), Fibrosis (0.72), Hernia (0.91), Infiltration (0.68), Mass (0.80), Nodule (0.69), Pleural Thickening (0.74), Pneumonia (0.71), Pneumothorax (0.75) |
| **Google Chest X-ray** | 3 | **0.84** | Lung Opacity (0.92), Fracture (0.74), Pneumothorax (0.85) |
| **RSNA Pneumonia** | 2 | **0.87** | Lung Opacity (0.88), Pneumonia (0.86) |
| **SIIM Pneumothorax** | 1 | **0.79** | Pneumothorax (0.79) |
| **PadChest** | 15 | **0.85** | Atelectasis (0.77), Cardiomegaly (0.93), Consolidation (0.88), Edema (0.97), Effusion (0.95), Emphysema (0.87), Fibrosis (0.94), Fracture (0.70), Hernia (0.96), Infiltration (0.85), Mass (0.85), Nodule (0.69), Pleural Thickening (0.79), Pneumonia (0.82), Pneumothorax (0.81) |
| **VinBrain** | 8 | **0.86** | Atelectasis (0.67), Cardiomegaly (0.90), Consolidation (0.93), Effusion (0.87), Infiltration (0.86), Lung Opacity (0.85), Pleural Thickening (0.84), Pneumothorax (0.93) |
| **CheXpert** | 11 | **0.86** | Atelectasis (0.91), Cardiomegaly (0.91), Consolidation (0.90), Edema (0.92), Enlarged Cardiomediastinum (0.78), Fracture (0.74), Lung Lesion (0.84), Lung Opacity (0.87), Effusion (0.94), Pneumonia (0.84), Pneumothorax (0.85) |
| **MIMIC-CXR** | 11 | **0.85** | Atelectasis (0.88), Cardiomegaly (0.88), Consolidation (0.91), Edema (0.92), Enlarged Cardiomediastinum (0.84), Fracture (0.74), Lung Lesion (0.82), Lung Opacity (0.86), Effusion (0.92), Pneumonia (0.82), Pneumothorax (0.81) |

---

## Model Configuration

| Attribute | Value |
|---|---|
| Model Name | DenseNet121-Res224-All |
| Framework | TorchXRayVision |
| Parameters | 6.97 Million |
| Architecture | DenseNet121 |
| Evaluation Metric | Area Under ROC Curve (AUC) |
| Task | Multi-label Chest X-ray Classification |

---

## Discussion

The DenseNet121-Res224-All model demonstrates strong cross-dataset generalization across diverse public chest radiography benchmarks. The model consistently achieves high AUC scores for clinically significant abnormalities including edema, cardiomegaly, consolidation, pleural effusion, and pneumothorax.

The highest overall performance is observed on the RSNA, VinBrain, CheXpert, and MIMIC-CXR datasets, with mean AUC values ranging between 0.85 and 0.87. Strong performance across heterogeneous datasets indicates robust feature learning and transferability for real-world medical imaging applications.

Performance degradation on subtle findings such as fibrosis, fractures, and nodules suggests these abnormalities remain comparatively difficult for automated chest X-ray interpretation systems.

---

## Source

Benchmarks obtained from TorchXRayVision official evaluation results. :contentReference[oaicite:0]{index=0}
