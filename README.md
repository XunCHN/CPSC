# Codes of Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration

Here is the official PyTorch implementation of *"Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration"*.

## Requirements

To run this code, please ensure your environment meets the following primary dependencies. The code has been tested with **CUDA Version: 12.1**.

* `torch==2.9.0`
* `torchvision==0.24.0`
* `torchaudio==2.9.0`
* `transformers==4.57.1`
* `timm==1.0.21`

*(Tip: You can install these core dependencies via `pip` or `conda` matching your CUDA 12.1 environment.)*

## Code Instruction

### Data Preparation

This repository uses the **CREMA-D**, **NYU Depth V2**, **SUN RGB-D** and **MVSA** datasets as examples. Please download and prepare the datasets from the following open-source repositories before running the code:

* **NYU Depth V2**, **SUN RGB-D** and **MVSA**: The processed datasets can be found in the [QMF repository](https://github.com/QingyangZhang/QMF/tree/main).
* **CREMA-D**: This dataset is available in the [NeurIPS24-LFM repository](https://github.com/njustkmg/NeurIPS24-LFM).

*(Note: After downloading, please ensure the datasets are placed in the appropriate data directories as expected by the dataloaders.)*

### Run

#### 1. CREMA-D

This experiment performs multimodal emotion recognition on the CREMA-D dataset.

**Step 1**: Modify the dataset path in [dataset/CREMA.py]. Set `data_root` in the `config` dictionary to your CREMA-D dataset root directory, which should contain `AudioWAV/` and `Image/` subdirectories.

**Step 2**: Run the training script:

```bash
python CPSC_cremad.py
```

#### 2. MVSA

This experiment performs multimodal sentiment analysis on the MVSA dataset.

**Step 1**: Download the pre-trained BERT model (`bert-base-uncased`) from the [QMF repository](https://github.com/QingyangZhang/QMF/tree/main) and place it under the `bert-base-uncased` folder at the project root.

**Step 2**: Download the `MVSA_Single` dataset from the [QMF repository](https://github.com/QingyangZhang/QMF/tree/main) and place it under the `datasets` folder at the project root.

**Step 3 (Training)**: Adjust the dataset path (`--data_path`), BERT model path (`--bert_model`), and model save path (`--savedir`) in [CPSC_MVSA.py]. Then run:

```bash
python CPSC_MVSA.py
```

**Step 4 (Robustness Testing)**: Modify the `--model_path` argument in [test.py] to point to your trained model checkpoint. Then run:

```bash
python test.py
```

This will evaluate the model under multiple noise scenarios (Clean, Gaussian noise at levels 5.0/10.0, and Salt & Pepper noise at levels 5.0/10.0).

A demo checkpoint (`MVSA_demo.pt`) is available for quick testing:
- **Baidu Netdisk**: [Link](https://pan.baidu.com/s/1QUzP75boVnk3hgSKTAmEPQ?pwd=ti42) | Password: `ti42`

#### 3. NYU Depth V2 & SUN RGB-D

These experiments perform RGB-D scene recognition on the NYU Depth V2 and SUN RGB-D datasets.

**Step 1**: Download the pre-trained ResNet-18 model checkpoint from the [QMF repository](https://github.com/QingyangZhang/QMF/tree/main) and place it under the `checkpoint` folder as `resnet18_pretrained.pth`.

**Step 2 (NYU Depth V2)**:

Adjust the dataset path (`--data_path`), model save path (`--savedir`), and checkpoint path (`--CONTENT_MODEL_PATH`) in [CPSC_nyu.py]. Then run:

```bash
python CPSC_nyu.py
```

**Step 3 (SUN RGB-D)**:

Adjust the dataset path (`--data_path`), model save path (`--savedir`), and checkpoint path (`--CONTENT_MODEL_PATH`) in [CPSC_sun.py]. Then run:

```bash
python CPSC_RGB/CPSC_sun.py
```


## Acknowledgements

We sincerely thank the authors of the following open-source repositories. Their excellent codebases greatly inspired and assisted our implementation:

* [QMF](https://github.com/QingyangZhang/QMF/tree/main)
* [NeurIPS24-LFM](https://github.com/njustkmg/NeurIPS24-LFM)
* [MMPareto_ICML](https://github.com/GeWu-Lab/MMPareto_ICML2024)
