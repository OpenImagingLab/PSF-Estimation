<div style="
background: linear-gradient(135deg, #007CF0, #7928CA);
-webkit-background-clip: text;
background-clip: text;
color: transparent;
font-size: 48px;
font-weight: bold;
font-family: Arial, sans-serif;
text-align: center;
margin: 40px 0;
display: inline-block;
padding: 10px;
">
A Physics-Informed Blur Learning Framework for Imaging Systems
</div>
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
|   main_two_step.py
|
|---configs
|   |   lensname.yaml
|
|---sfrmat5_dist
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

## Training
To train an aberration learning model from scratch, run `main.py`. The results will be saved in /results/lensname.