# AgriAlignNER-Disease7000-Refined
Reproducibility resources for Disease7000-Refined, including data-cleaning and reconstruction scripts, identifiers of 6,298 retained samples, and fixed train/validation/test split files.


# AgriAlignNER Reproducibility Package

This repository provides the reproducibility materials associated with the AgriAlignNER study and the Disease7000-Refined dataset.

The raw images, texts, and annotations used to construct Disease7000-Refined originate from the dataset developed by Zhang et al., entitled *Chinese Named Entity Recognition for Agricultural Diseases Based on Entity-Related Visual Prompts Injection*, and remain subject to the original data usage agreement. Therefore, the raw dataset is not redistributed in this repository.

Researchers who obtain the original dataset through the authorized channel can use the released sample identifiers, fixed split files, and reconstruction script to reproduce the exact dataset version used in this study.

## Repository Contents

```
AgriAlignNER-Reproducibility/
├── README.md
├── reconstruct_refined_dataset.py
│
├── manifests/
│   └── retained_sample_ids.txt
│
└── splits/
    ├── train_ids.txt
    ├── valid_ids.txt
    └── test_ids.txt
```

The released files include:

* `reconstruct_refined_dataset.py`: validates the released sample identifiers and reconstructs the fixed multimodal data splits from an authorized local copy of the original dataset.
* `manifests/retained_sample_ids.txt`: contains the identifiers of all 6,298 samples retained after data cleaning.
* `splits/train_ids.txt`: contains the 5,038 training-sample identifiers.
* `splits/valid_ids.txt`: contains the 629 validation-sample identifiers.
* `splits/test_ids.txt`: contains the 631 test-sample identifiers.

The three fixed splits are mutually exclusive, and their union is identical to the set of retained sample identifiers.

## Obtaining the Original Data

The original images, texts, and annotations are not included in this repository.

Researchers should obtain the original dataset developed by Zhang et al., entitled *Chinese Named Entity Recognition for Agricultural Diseases Based on Entity-Related Visual Prompts Injection*, in accordance with its original data usage conditions. After obtaining authorized access, place the original files in the following local directories:

```
data/
├── Disease7000-Refined/
├── Disease7000-Refined_images/
└── Disease7000-Refined_aux_images/
```

The expected main-image naming format is:

```
data/Disease7000-Refined_images/1.jpg
data/Disease7000-Refined_images/2.jpg
```

The expected auxiliary-image naming format is:

```
data/Disease7000-Refined_aux_images/train/crops/1_pred_yolo_crop_0.png
data/Disease7000-Refined_aux_images/train/crops/1_pred_yolo_crop_1.png
```

Auxiliary images for the validation and test sets should be placed under the corresponding `valid/crops/` and `test/crops/` directories.

The textual annotation files should contain samples beginning with an identifier line such as:

```text
IMGID:1093
Tomato	B-Crop
leaves	I-Crop
suffering	O
from	O
tomato	B-Disease
early	I-Disease
blight	I-Disease
have	O
small	B-Feature
black	I-Feature
elliptical	I-Feature
spots	I-Feature
at	B-Position
the	I-Position
leaf	I-Position
tips	I-Position
```

## Sample-Identifier Format

All released identifier files use the following format:

```
IMGID:31
IMGID:1093
IMGID:5216
```

Each `IMGID` uniquely identifies one multimodal sample, including:

* its textual content;
* its BIO annotation sequence;
* its corresponding main image;
* its associated auxiliary images, when available.

The same identifier is used to ensure that all modalities belonging to one sample are assigned to the same data split.

## Environment

The reconstruction script requires Python 3.8 or later and uses only Python standard-library modules.


## Reconstruction

Run the script from the repository root:

```bash
python reconstruct_refined_dataset.py
```

If the output directory already exists and should be replaced, run:

```bash
python reconstruct_refined_dataset.py --overwrite
```


## Generated Files

After successful execution, the following files and directories are generated locally:

```
manifests/
├── retained_sample_ids.txt
└── reconstruction_report.txt

reconstructed_data/
├── Disease7000-Refined/
│   ├── train.txt
│   ├── valid.txt
│   └── test.txt
│
├── Disease7000-Refined_images/
│   ├── train/
│   ├── valid/
│   └── test/
│
└── Disease7000-Refined_aux_images/
    ├── train/
    │   └── crops/
    ├── valid/
    │   └── crops/
    └── test/
        └── crops/
```

The script uses the same `IMGID` files to reconstruct the textual annotations, main images, and auxiliary images, ensuring that all modalities of one sample remain in the same fixed split.

## Expected Split Statistics

A successful reconstruction should produce:

| Split | Number of samples |
|-------| ----------------: |
| Train |             5,038 |
| Valid |               629 |
| Test  |               631 |
| Total |             6,298 |

The reconstruction script also verifies that:

* no identifier is repeated within a split;
* no identifier appears in more than one split;
* the union of the three splits contains exactly 6,298 identifiers;
* all retained identifiers have corresponding textual annotations;
* all retained identifiers have corresponding main images.

Samples without auxiliary images are reported but do not automatically terminate reconstruction, because auxiliary-image availability may depend on the preprocessing configuration.

## Data Redistribution

The following materials are intentionally excluded from this repository:

```
data/Disease7000-Refined/
data/Disease7000-Refined_images/
data/Disease7000-Refined_aux_images/
reconstructed_data/
```

These directories contain or may contain data derived from the original dataset and should not be publicly redistributed without permission from the original data provider.

## Reproducibility Scope

This repository enables researchers with authorized access to the original dataset to reproduce:

* the complete set of retained samples;
* the fixed training, validation, and test split;
* the correspondence among textual annotations, main images, and auxiliary images;
* the sample counts reported in the paper.

