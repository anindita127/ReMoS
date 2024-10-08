# ReMoS: 3D Motion-Conditioned Reaction Synthesis for Two-Person Interactions 
## Accepted at the European Conference on Computer Vision (ECCV) 2024.

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
Set 'is_eval' flag to True to get the evaluation metrics.
