# ReMoS: 3D Motion-Conditioned Reaction Synthesis for Two-Person Interactions 
Accepted at the European Conference on Computer Vision (ECCV) 2024.

[Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05358.pdf) | 
[Video](https://vcai.mpi-inf.mpg.de/projects/remos/Remos_ECCV_v2_1.mp4) | 
[Project Page](https://vcai.mpi-inf.mpg.de/projects/remos/)

<img src="https://vcai.mpi-inf.mpg.de/projects/remos/images/teaser.jpg" alt="teaser image" />



## Pre-requisites
We have tested our code on the following setups: 
* Ubuntu 20.04 LTS
* Windows 10, 11
* Python >= 3.8
* Pytorch >= 1.11
* conda >= 4.9.2 (optional but recommended)

## Getting started

Follow these commands to create a conda environment:
```
conda create -n remos python=3.8
conda activate remos
conda install -c pytorch pytorch=1.11 torchvision cudatoolkit=11.3
pip install -r requirements.txt
```
For pytorch3D installation refer to https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

**Note:** If PyOpenGL installed using `requirements.txt` causes issues in Ubuntu, then install PyOpenGL using:
```
apt-get update
apt-get install python3-opengl
```

## Dataset download and preprocess
Download the ReMoCap dataset from the [ReMoS website](https://vcai.mpi-inf.mpg.de/projects/remos/#dataset_section). Unzip and place th dataset under `../DATASETS/ReMoCap`. 
The format of the dataset folder should be as follows:
```bash
    DATASETS
    ├── ReMoCap
    │   │
    │   ├── LindyHop
    │       │
    │       ├── train
    │              │
    │              └── seq_3
    │                   │
    │                   └── 0 'first person'
    │                       └── motion.bvh
    │                       └── motion_worldpose.csv
    │                       └── motion_rotation.csv
    │                       └── motion_offsets.pkl
    │                   └── 1 'second person'
    │                       └── motion.bvh
    │                       └── motion_worldpose.csv
    │                       └── motion_rotation.csv
    │                       └── motion_offsets.pkl
    │         
    │              └── ... 
    │       ├── test
    │              │
    │              └── ...
    |
    │   ├── Ninjutsu
    │       │
    │       ├── train
    │              │
    │              └── shot_001
    │                   │
    │                   └── 0.bvh
    │                   └── 0_worldpose.csv
    │                   └── 0_rotations.csv
    │                   └── 0_offsets.pkl
    │                   └── 1.bvh
    │                   └── 1_worldpose.csv
    │                   └── 1_rotations.csv
    │                   └── 1_offsets.pkl
    │              └── shot_002
    │                   └── ...
    │              └── ... 
    │       ├── test
    │              │
    │              └── ...

```

3. To pre-process the two parts of the dataset for our setting, run: 
```
python src/Lindyhop/process_LindyHop.py
python src/Ninjutsu/process_Ninjutsu.py
```
This will create the 'train.pkl' and 'test.pkl' under `data/` folder.

## Training and testing on the Lindy Hop motion data 

4. To train the ReMoS model on the Lindy Hop motions in our setting, run:
```
python src/Lindyhop/train_body_diffusion.py
python src/Lindyhop/train_hand_diffusion.py
```

5. To test and evaluate the ReMoS model on the Lindy Hop motions, run:
```
python src/Lindyhop/test_full_diffusion.py
```
Set 'is_eval' flag to True to get the evaluation metrics, and set 'is_eval' to False to visualize the results.

Download the pre-trained weights for the Lindy Hop motions from [here](https://vcai.mpi-inf.mpg.de/projects/remos/LindyHop_pretrained_weights.zip) and unzip them under `save/LindyHop/`.
 ** Training codes and pre-trained weights for Ninjutsu motions coming soon! **

## License

Copyright (c) 2024, Max Planck Institute for Informatics 
All rights reserved.

Permission is hereby granted, free of charge, to any person or company obtaining a copy of this dataset and associated documentation files (the "Dataset") from the copyright holders to use the Dataset for any non-commercial purpose. Redistribution and (re)selling of the Dataset, of modifications, extensions, and derivates of it, and of other dataset containing portions of the licensed Dataset, are not permitted. The Copyright holder is permitted to publically disclose and advertise the use of the software by any licensee.

Packaging or distributing parts or whole of the provided software (including code and data) as is or as part of other datasets is prohibited. Commercial use of parts or whole of the provided dataset (including code and data) is strictly prohibited.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE DATASET.


