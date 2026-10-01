# AgriFuseNER* (Author Reimplementation)

This directory contains the author reimplementation of AgriFuseNER used for the controlled comparison reported in the revised manuscript.

The implementation was developed according to the methodological description in the original paper:

**Entity-level cross-modal fusion for multimodal chinese agricultural diseases and pests named entity recognition**

This is not the official implementation released by the original authors. Because some implementation details are not publicly available, this reimplementation follows the methodological description in the paper and the experimental settings documented below.

## Requirements

To run the code, you need to install the requirements:
```
pip install -r requirements.txt
```

## Pretrained Models

### BERT Text Encoder

The textual encoder is initialized with `bert-base-cased` using the Hugging Face Transformers library.

The pretrained BERT model and tokenizer are automatically downloaded and cached on first use. Therefore, no manually downloaded local BERT checkpoint is required.

### ViT Visual Encoder

The visual encoder is initialized with`google/vit-base-patch16-224` using the Hugging Face Transformers library.

The pretrained ViT model and image processor are automatically downloaded and cached on first use. Therefore, no manually downloaded local ViT checkpoint is required.

## Dataset

AgriFuseNER* is evaluated on Disease7000-Refined dataset:

Disease7000-Refined is an agricultural multimodal named entity recognition dataset reconstructed from an existing agricultural disease dataset.

Due to the original data usage agreement, the raw images, texts, and annotations cannot be publicly redistributed.

The dataset directory should be organized as follows:
```
AgriFuseNER*
 |-- data
 |    |-- Disease7000-Refined
 |    |    |-- train.txt
 |    |    |-- valid.txt
 |    |    |-- test.txt
 |    |    |-- Disease7000-Refined_train_dict.pth
 |    |    |-- Disease7000-Refined_val_dict.pth
 |    |    |-- Disease7000-Refined_test_dict.pth
 |    |-- Disease7000-Refined_images
 |    |-- Disease7000-Refined_aux_images
 |    |    |-- train
 |    |    |    |-- crops
 |    |    |-- val
 |    |    |    |-- crops
 |    |    |-- test
 |    |    |    |-- crops
 |    |-- agrifusener_vit_features
 |    |    |-- train
 |    |    |-- dev
 |    |    |-- test
 |-- models
 |    |-- __init__.py
 |    |-- agrifusener.py
 |-- modules
 |    |-- __init__.py
 |    |-- crf.py
 |-- processor
 |    |-- __init__.py
 |    |-- dataset.py
 |-- utils
 |    |-- __init__.py
 |    |-- evaluate.py
 |    |-- visual.py
 |-- README.md
 |-- requirements.txt
 |-- run_agrifusener.py
```

The directory:

```text
data/agrifusener_vit_features/
```
is automatically generated during the visual feature-extraction stage and is not required to be included in the released repository.

Before training, run:

```bash
python run_agrifusener.py --mode extract
```

The extracted features are stored under:

```text
data/agrifusener_vit_features/
```

with separate directories for the training, validation, and test sets.

The feature files are generated locally and are not included in the repository.

## Running the Code

All commands below should be executed from the `AgriFuseNER/` directory.

### 1. Validate the Dataset

Before feature extraction and training, verify the dataset and fixed split sizes:

```bash
python run_agrifusener.py --mode validate
```

The expected number of samples is:

```text
Train: 5038
Valid: 629
Test:  631
```

### 2. Extract ViT Visual Features

Run:

```bash
python run_agrifusener.py --mode extract
```

This step extracts ViT representations from the global images and the top three YOLOv11s-generated ROI crops.

If the pretrained ViT model is not already cached, it will be automatically downloaded from Hugging Face.

### 3. Training

#### Training on Disease7000-Refined

```bash
python run_agrifusener.py \
  --mode train \
  --lambda-paper 0.1 \
  --grouping-source-train predicted \
  --seed 2021 \
  --out outputs/agrifusener_seed2021
```

The best checkpoint is selected according to validation F1 and saved as:

```text
outputs/agrifusener_seed2021/best.pt
```

### 4. Testing

#### Testing on Disease7000-Refined
```bash
python run_agrifusener.py \
  --mode test \
  --checkpoint outputs/agrifusener_seed2021/best.pt \
  --lambda-paper 0.1 \
  --grouping-source-train predicted \
  --seed 2021 \
  --out outputs/agrifusener_seed2021_test
```

## Notes on Reproducibility

AgriFuseNER* denotes the author reimplementation used in the revised manuscript and should not be interpreted as the official implementation of the original AgriFuseNER paper.




