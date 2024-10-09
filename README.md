## AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion

#### Yitong Jiang, Zhaoyang Zhang, Tianfan Xue, Jinwei Gu

The official PyTorch implementation of the paper **[AutoDIR: Automatic All-in-One Image Restoration with Latent Diffusion]**

## News: Accepted by ECCV2024

| [Github](https://github.com/jiangyitong/AutoDIR) |  [Page](https://jiangyitong.github.io/AutoDIR_webpage/) |  [Arxiv](https://arxiv.org/abs/2310.10123) | 

<a href="https://colab.research.google.com/drive/1tnBnBOSUqJvLqJqG1rWM_R0L6qc1UtW9?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

We propose an all-in-one image restoration system with latent diffusion, named AutoDIR, which can automatically detect and restore images with multiple unknown degradations. 
Our main hypothesis is that many image restoration tasks, such as super-resolution, motion deblur, denoising, low-light enhancement, dehazing, and deraining can often be decomposed into some common basis operators which improve image quality in different directions. AutoDIR aims to learn one unified image restoration model capable of performing these basis operators by joint training with multiple image restoration tasks.
Specifically, AutoDIR consists of a Blind Image Quality Assessment (BIQA) module based on CLIP which automatically detects unknown image degradations for input images, an All-in-One Image Restoration (AIR) module based on latent diffusion which handles multiple types of image degradation, and a Structural-Correction Module (SCM) which further recovers the image structures. 
Extensive experimental evaluation demonstrates that AutoDIR outperforms state-of-the-art approaches for a wider range of image restoration tasks. The design of AutoDIR also enables flexible user control (via text prompt) and generalization to new tasks as a foundation model of image restoration.  

## Demo Video
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/WZJhS-Qo6TA/0.jpg)](https://www.youtube.com/watch?v=WZJhS-Qo6TA)

## Set up a conda environment
```
conda env create -f environment.yaml
conda activate autodir
cd NAFNet
python setup.py develop --no_cuda_ext
pip install git+https://github.com/openai/consistencydecoder.git
```

## Updates
- **2023.01.19**: Add Colab demo of AutoDIR. <a href="https://colab.research.google.com/drive/1tnBnBOSUqJvLqJqG1rWM_R0L6qc1UtW9?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
- **2024.01.19**: Inference code and [pre-trained models](https://drive.google.com/drive/folders/1hVUyHY9FOUsEpFhPQPHmyd8diDNThqaU?usp=sharing) are released.

### TODO
- [ ] With Latent-Consistency-model
- [ ] Code for training from scratch
- [ ] Code for finetuning to new image restoration tasks
- [x] ~~CodeLab release~~
- [x] ~~Pre-trained model release~~
- [x] ~~Inference code release~~


## Sections.

### 1. Inference
***In this section, we provide the script to evaluate automatically handling images with unknown degradations and use-customization image restoration.***

#### easy start <a href="https://colab.research.google.com/drive/1tnBnBOSUqJvLqJqG1rWM_R0L6qc1UtW9?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>

#### Handling images with unknown degradations iteratively
```
python eval_w_SCM.py
--input
test_images\unknown.png
--steps
200
--output
\path\to\our\output_folder
--ckpt
\path\to\the\pretrained\checkpoints
--cfg-text
1
--config
"configs/generate.yaml"
```
#### Example of open-vocabulary user control
![examply of open-vocabulary user control](https://github.com/jiangyitong/AutoDIR/blob/main/figs/user_control.png?raw=true)
#### Example of multiple unknown artifacts
![examply of multiple artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/multiple.png?raw=true)
#### Example of low-resolution artifact
![examply of low-resolution artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/sr.png?raw=true)
#### Example of blur artifact
![examply of blur artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/blur.png?raw=true)
#### Example of noise artifact
![examply of noise artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/noise.png?raw=true)
#### Example of raindrop artifact
![examply of raindrop artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/raindrop.png?raw=true)
#### Example of rain artifact
![examply of rain artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/rain.png?raw=true)
#### Example of haze artifact
![examply of haze artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/haze.png?raw=true)
#### Example of underexposure artifact
![examply of underexposure artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/lol.png?raw=true)


#### User-customized image restoration
```
python eval_w_SCM.py
--customize
"A photo needs underexposure artifact reduction"
--input
test_images\noise.png
--steps
200
--output
\path\to\our\output_folder
--ckpt
\path\to\the\pretrained\checkpoints
--cfg-text
1
--config
"configs/generate.yaml"
```
#### Example of user-customized image restoration
![examply of user-customized artifacts](https://github.com/jiangyitong/AutoDIR/blob/main/figs/user_customize.png?raw=true)

**Result structure:**

```
results/
    └── imageName_input.png   #input
    └── imageName_result.png   #result of All-in-one image restoration(Air) Module
    └── imageName_result_w_SCM.png   #result with Structure-correction-module(SCM)
    └── imageName_result_colorcorrect.png   #Air Module result after color correction (if appliable)
    └── imageName_result_w_SCM_colorcorrect.png   #Result after SCM after color correction (if appliable)
```


## Comments
This implementation is based on [StableDiffusion](https://github.com/CompVis/stable-diffusion), [NAFNet](https://github.com/megvii-research/NAFNet), [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix), [stablesr](https://github.com/IceClear/StableSR)

## Citation
If you find our paper useful for your research, please consider citing our work :blush: : 
```
@article{jiang2023autodir,
  title={Autodir: Automatic all-in-one image restoration with latent diffusion},
  author={Jiang, Yitong and Zhang, Zhaoyang and Xue, Tianfan and Gu, Jinwei},
  journal={arXiv preprint arXiv:2310.10123},
  year={2023}
}
```

