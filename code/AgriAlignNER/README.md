# Agricultural Multimodal Named Entity Recognition via Hierarchical Entity-Token Cross-Modal Alignment

This repository contains the source code for the paper:

**Agricultural Multimodal Named Entity Recognition via Hierarchical Entity-Token Cross-Modal Alignment**

Requirements
==========
To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

Dataset
==========
AgriAlignNER is evaluated on two multimodal named entity recognition datasets:
1. Disease7000-Refined
2. Twitter2017

一. Disease7000-Refined

Disease7000-Refined is an agricultural multimodal named entity recognition dataset reconstructed from an existing agricultural disease dataset.

Due to the original data usage agreement, the raw images, texts, and annotations cannot be publicly redistributed. The preprocessing scripts, retained sample identifiers, and fixed training/validation/test splits are provided in this repository for reproducibility.

The dataset directory should be organized as follows:
```
AgriAlignNER
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
 |    |    |-- val
 |    |    |-- test
 |-- models
 |    |-- agri_align_model.py
 |-- modules
 |    |-- agri_align_trainer.py
 |-- utils
 |    |-- encoder.py
 |-- run_agriAlign.py
 
```
二. Twitter2017
Twitter2017 is a widely used benchmark dataset for multimodal named entity recognition.
The text data follows the conll format. You can download the Twitter2017 data via this [link](https://drive.google.com/file/d/1ogfbn-XEYtk9GpUECq1-IwzINnhKGJqy/view?usp=sharing).
The dataset directory should be organized as follows:
```
AgriAlignNER
 |-- data
 |    |-- Twitter2017
 |    |    |-- train.txt
 |    |    |-- valid.txt
 |    |    |-- test.txt
 |    |    |-- Twitter2017_train_dict.pth
 |    |    |-- Twitter2017_val_dict.pth
 |    |    |-- Twitter2017_test_dict.pth
 |    |-- Twitter2017_images
 |    |-- Twitter2017_aux_images
 |    |    |-- train
 |    |    |-- val
 |    |    |-- test
 |-- models
 |    |-- agri_align_model.py
 |-- modules
 |    |-- agri_align_trainer.py
 |-- utils
 |    |-- encoder.py
 |-- run_agriAlign.py
 
```

## Run

### Training

#### Training on Disease7000-Refined

```bash
python run_agriAlign.py \
--dataset_name Disease7000-Refined \
--bert_name bert-base-cased \
--use_prompt \
--use_amgca \
--do_train \
--num_epochs 60 \
--batch_size 32 \
--lr 3e-6 \
--warmup_ratio 0.05 \
--max_seq 256 \
--save_path ./checkpoints
```

#### Training on twitter2017
```bash
python run_agriAlign.py \
--dataset_name Twitter2017 \
--bert_name bert-base-cased \
--use_prompt \
--use_amgca \
--do_train \
--num_epochs 60 \
--batch_size 32 \
--lr 3e-6 \
--warmup_ratio 0.05 \
--max_seq 256 \
--save_path ./checkpoints
```

### Testing

#### Testing on Disease7000-Refined
```bash
python run_agriAlign.py \
--dataset_name Disease7000-Refined \
--bert_name bert-base-cased \
--use_prompt \
--use_amgca \
--only_test \
--load_path ./checkpoints/best_model.pt \
--batch_size 32
```

#### Testing on twitter2017
```bash
python run_agriAlign.py \
--dataset_name Twitter2017 \
--bert_name bert-base-cased \
--use_prompt \
--use_amgca \
--only_test \
--load_path ./checkpoints/best_model.pt \
--batch_size 32
```


## Acknowledgement

The preprocessing procedure for Twitter2017 follows the implementation provided by UMT:

https://github.com/jefferyYu/UMT/

We sincerely thank the authors for releasing their multimodal named entity recognition resources.




