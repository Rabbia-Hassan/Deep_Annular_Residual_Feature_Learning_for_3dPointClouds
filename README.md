# Deep_Annular_Residual_Feature_Learning_for_3dPointClouds

We propose a novel architecture for 3d point cloud classification and part segmentation.
![Teaser](/images/teaser.svg)
1. [Introduction](#introduction)
2. [Architecture](#Architecture)
3. [Setup](#setup)
4. [Testing](#Usage)
5. [Acknowledgements](#acknowledgements)
## Introduction
We propose a deep, hierarchichal 3d point based architecture for raw point cloud processing which employes an enhanced version 
of Annular Convolution and Residual Learning Principle to integrate features along the hierarchy.
## Architecture
![Proposed Architecture](/images/arch_diagram_svg1.svg)
 ## Setup
 We provide the code of proposed network, which is tested with following configuration.
* Tensorflow gpu version 1.15.0
* CUDA 10.0
* GCC compiler 7.5.0
* Python 2.7
* Ubuntu 18.04

All experiments are performed on a single NVIDIA TITAN Xp GPU with 12 GB GDDR5X equipped server.

The customized TensorFlow operators are included in the folder “tf_ops”. There are four
customized operators in total. To compile these operators please refer to [Pointnet++](https://github.com/charlesq34/pointnet2) for guidance.
## Testing

## Object Classification (synthetic dataset)
To train the Classification model from scratch download the [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip) dataset, unzip it and place it in the folder named as data.
To start the training type the following command.
```
python train.py
```
To evaluate the trained model, type the following command.
```
python evaluate.py
```
To do 20 evaluations and keep the best result, run the  script given below which will write the
log of all 20 evaluations in a file named as “Record.csv”.
```
python script_classification.py
```

## Object Classification (realworld dataset)

Firstly, navigate to the directory named as SONN from the main directory using following
command
```
cd SONN
```
To train the Classification model from scratch download the [ScanobjectNN](https://drive.google.com/drive/folders/1yxX-IeYSmIEoKtUvCx4neFqhb3Csd2jq?usp=sharing) dataset. It’s a pre-processed version of the dataset with 
surface normals which have been computed using Least square plane estimation method. Unzip it and place it in the folder named as data.

To initiate the training type the following command
```
python train.py
```
To evaluate the trained model run the following script.
```
python evaluate.py
```


## Object Part Segmentation
Navigate to the directory named as part_segm using following command.

```
cd part_segm
```

To train the model of part Segmentation from scratch, download the [ShapeNet part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) dataset comprised of XYZ, normal and part labels dataset unzip it and place it in the folder named as data.


To train the model, run the following script.
```
python train.py
```
To evaluate the trained model run the following script.
```
python evaluate.py
```
Trained models for all three datasets can be found [here](https://drive.google.com/drive/folders/1D6vT-kEZ2ewZTAJpLWqbNix33wwVziP9?usp=sharing)

## Acknowledgements
The code for training and evaluation is borrowed from [Pointnet++](https://github.com/charlesq34/pointnet2) and that of Annular Convolution is borrowed from [ACNN](https://github.com/artemkomarichev/a-cnn.git)

## Citation
If you find our work useful in your research, please cite our work:

@article{hassan2023residual,
  title={Residual Learning with Annularly Convolutional Neural Networks for Classification and Segmentation of 3D Point Clouds},
  author={Hassan, R and Fraz, MM and Rajput, A and Shahzad, M},
  journal={Neurocomputing},
  year={2023},
  publisher={Elsevier}
}
