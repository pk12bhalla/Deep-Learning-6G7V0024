# Deep-Learning-6G7V0024
# Concrete Crack Detection using Deep Learning

This project was built as part of my Deep Learning module (6G7V0024) at Manchester Metropolitan University. The idea was pretty straightforward - can we train a model to look at a photo of a concrete surface and tell whether it has a crack or not?

Turns out, yes. And it works really well.

---

## What this project does

I trained three different models on 40,000 images of concrete surfaces (half cracked, half not) and compared how they performed:

- **CrackCNN** - a custom CNN I built from scratch
- **ResNet-18** - a standard pretrained model used as a baseline
- **EfficientNet-B0** - a more efficient pretrained model for comparison

The best result was ResNet-18 hitting **99.88% test accuracy**, but honestly all three models performed well above 99%, which was better than I expected going in.

I also built a small Gradio web app at the end so anyone can upload an image and get a prediction without touching any code.

---

## Results summary

| Model | Test Accuracy | AUC-ROC |
|---|---|---|
| CrackCNN (custom) | 99.63% | 0.9996 |
| ResNet-18 | 99.88% | 0.9999 |
| EfficientNet-B0 | 99.78% | 0.9999 |

---

## Dataset

I used the **Surface Crack Detection** dataset by Özgenel & Sorguç (2018), available on Kaggle:

https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

It has 40,000 RGB images split equally between two classes - cracked and not cracked. Download it from Kaggle and set it up like this:

```
data/
└── Concrete Crack Images/
    ├── Positive/   ← cracked surfaces
    └── Negative/   ← intact surfaces
```

---

## How to run it

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/concrete-crack-detection.git
cd concrete-crack-detection
```

**2. Install dependencies**
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas gradio kagglehub
```

**3. Download the dataset**

Either manually from Kaggle, or add this at the top of the notebook:
```python
import kagglehub
path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
```

**4. Open and run the notebook**
```bash
jupyter notebook PrashantKumarBhalla_25948685_Notebook.ipynb
```

Run all cells from top to bottom. Training takes around 20-30 minutes on a GPU (I used Google Colab with a T4).

---

## What's inside the notebook

The notebook walks through everything step by step:

1. Setup and imports
2. Dataset loading and exploratory analysis
3. Data preprocessing and augmentation
4. Model definitions - CrackCNN, ResNet-18, EfficientNet-B0
5. Training all three models
6. Evaluation - accuracy, precision, recall, F1, AUC-ROC
7. Confusion matrices and ROC curves
8. Qualitative results (correct vs incorrect predictions)
9. Grad-CAM visualisations (showing where the model looks)
10. Ablation study - with vs without data augmentation
11. Hyperparameter tuning - comparing learning rates
12. Gradio web app for real-time predictions
13. Final summary

---

## Requirements

- Python 3.10+
- PyTorch 2.x
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas
- gradio
- kagglehub

A GPU is strongly recommended. The notebook was developed and tested on Google Colab with a T4 GPU.

---

## Project structure

```
concrete-crack-detection/
├── PrashantKumarBhalla_25948685_Notebook.ipynb   ← main notebook
├── README.md                                     ← this file
```

Figures and model checkpoints are generated automatically when you run the notebook.

---

## Author

**Prashant Kumar Bhalla**  
MSc Data Science - Manchester Metropolitan University  
25948685
