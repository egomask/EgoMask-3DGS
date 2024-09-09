# Egocentric 4D Complex Scene Rendering and Hands Decomposition with Deformable 3D Gaussian Splatting

## [Project page](https://egomask.github.io/)

![Teaser image](pipline/Output.png)

This repository contains the official implementation associated with the paper "Egocentric 4D Complex Scene Rendering and Hands Decomposition with Deformable 3D Gaussian Splatting".



## Dataset

In our paper, we use:

- Craft Egocentric Dataset [AImove](https://www.kaggle.com/datasets/olivasbre/aimove?select=EgoMask).

We organize the datasets as follows:

```shell
├── data EgoMask
│   | glass 
│     ├── depth_gt
│     ├── images 
│     ├── masks
│     ├── ...
│   | leather
│     ├── depth_gt
│     ├── images 
│     ├── masks
│     ├── ...
│   | marble
│     ├── depth_gt
│     ├── images 
│     ├── masks
│     ├── ...
```

> Each frame is paired with a corresponding depth image and hands mask, generated by prior data-driven models,  [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2) and [Track Anything](https://github.com/gaomingqi/Track-Anything). Additionally, each frame includes paired camera space position, orientation, and extrinsic and intrinsic parameters generated by COLMAP.



## Pipeline

![Teaser image](pipline/Egomask.png)
We leverage 2D prior data, including hand masks and depth images, to guide the dynamic 3D Gaussian representation process. This approach enhances the capture of fine scene details and  improves the rendering performance of dynamic hands. 


## Overview 

The codebase has 2 main components: 
A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10

## Run

### Hardware Requirements 
- CUDA-ready GPU with Compute Capability 7.0+


## Software Requirements
Conda 
- C++ Compiler for PyTorch extensions, Visual Studio 2019 more stable (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install after Visual Studio (I used 11.6)
- C++ Compiler and CUDA SDK must be compatible

### Environment


```shell
git clone https://github.com/egomask/EgoMask-3DGS.git --recursive
cd EgoMask-3DGS

conda create -n EgoMask_env python=3.7
conda activate EgoMask_env

# install pytorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install dependencies
pip install -r requirements.txt
```



### Train

**EgoMask Dataset:**

```shell
python train.py -s path/to/Egomask/dataset -m output/exp-name --eval --is_blender
```

### Render & Evaluation

```shell
python render.py -m output/exp-name --mode render
python metrics.py -m output/exp-name
```






## Acknowledgments

Our code is developed based on [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [Deformable 3D Gaussians](https://ingra14m.github.io/Deformable-Gaussians/). Many thanks to the authors for open-sourcing the codebase.
