# **A Physics-Informed Blur Learning Framework for Imaging Systems**
<p align="center" style="font-size:18px;">
  <a href="https://arxiv.org/abs/2502.11382"><b>📜 Paper</b></a> &nbsp;  
  <a href="https://github.com/OpenImagingLab/PSF-Estimation"><b>💻 Code</b></a> &nbsp;  
  <a href="https://openimaginglab.github.io/PSF-Estimation/"><b>🌐 Project Page</b></a>
</p>



[//]: # (Liqun Chen, Yuyao Hu, Jiewen Nie, Tianfan Xue and Jinwei Gu)

## Environment requirements
The codes was tested on Windows 10, with Python and PyTorch. Required packages:
- numpy  
- tqdm
- python
- matplotlib
- torch
- torchvision
- pandas
- opencv-python
- pyyaml

## File structure
This repository contains codes for OAE(optical aberration estimation).
```
OAE
|   README.md
|   main.py
|   generate_edges_rgb.py
|   generate_fov_weight.py
|
|---configs
|   |   lensname.yaml
|
|---sfrmat5_dist
|   |---sfrmat5
|       |   user_sfrmat5_rgb.m
|
|---dataset 
|   |---lensname
|       |   npy
| 
|---input 
|   |   lensname.xlsx
| 
|---model 
|   |   optics_rgb.py
|   |   PSF_mlp.py
| 
|---results 
|
|---utils 
|   |   tools.py
|   |   train.py
```
`/model` contains the optical aberration model.

`/dataset` includes datasets used for training the optical aberration model.

`/sfrmat5_dist` contains the SFR calculation algorithm, which was downloaded from [ISO 12233](https://www.imaging.org/site/IST/Standards/Digital_Camera_Resolution_Tools/IST/Standards/Digital_Camera_Resolution_Tools.aspx#msw.).

`/results` stores the results, including the PSF map and PSF comparisons.

## SFR Data Preparation

Follow these steps to generate the necessary SFR (Spatial Frequency Response) data for training.

1.  **Generate Blurry Edges:**
    Run the Python script `generate_edges_rgb.py` to process your images and obtain the initial blurry edge data.
    ```bash
    python generate_edges_rgb.py
    ```

2.  **Calculate SFR Curves:**
    Next, open MATLAB and execute the `user_sfrmat5_rgb.m` script. This will analyze the edges from the previous step to compute and save the SFR curves.

3.  **Generate Training Data:**
    Finally, run the `generate_fov_weight.py` script. This will process the SFR curves and generate the final `.npz` files that will be used as input for the training process.
    ```bash
    python generate_fov_weight.py
    ```

## Training
To train an aberration learning model from scratch, run `main.py`. The results will be saved in /results/lensname.