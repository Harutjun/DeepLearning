# Interactive CycleGan - pytorch

<h1 align="center">
  <br>
Interactive CycleGan - Applied CycleGan architecture to create cool interactive apps
  <br>
</h1>
  <p align="center">
    Elay Dahan •
    Harutjun Magakyan
  </p>


<p align="center">
  <img src="https://github.com/Harutjun/DeepLearning/raw/main/assets/sketch2shoes.png" height="120">
</p>
<p align="center">
  <img src="https://github.com/Harutjun/DeepLearning/raw/main/assets/app_gif.gif" height="100">
</p>

# Interactive CycleGan

> **About this project:** *We implemented the CycleGan with pytorch framework.
CycleGan is a Generative adverserial netowrk used to learn style transfer between two image domain.
CycleGan is an conditional GAN which works directly on images from one domains.
Worth noticing the CycleGan, unline other Style transfer GAN's work's on unpaired images, i.e, images can be different both in content and style.
The CycleGan architecture consists of 4 neural networks, 2 Generator and 2 Discriminators.
Let X,Y be 2 image domains, the first pair of generator and discriminator is used 
To learn the mapping F(x) such that, F(x) has the same style as Y.
The discriminator is used to disriminate between the fake generated Y-like sample, F(x) and realy sample form Y domain, y.
The same is for the second pair of the generator and discriminator but with the opposite direction.
The full model architecture, loss function, dataset's and training procedure can be found in the original article: https://arxiv.org/pdf/1703.10593.pdf
We built an interactive app that can generate shoe images based on user drawing.
The model is trained on the Zappos50K datset.
We also applied this model to colorize Gray scale images to RGB images.*

  * [Citation](#citation)
  * [Prerequisites](#prerequisites)
  * [Repository Organization](#repository-organization)
  * [Credits](#credits)
    

## Prerequisites

* For your convenience, we provide an `environemnt.yml` file which installs the required packages in a `conda` environment name `torch`.
    * Use the terminal or an Anaconda Prompt and run the following command `conda env create -f environment.yml`.
* For Style-SoftIntroVAE, more packages are required, and we provide them in the `style_soft_intro_vae` directory.


|Library         | Version |
|----------------------|----|
|`Python`|  `3.6 (Anaconda)`|
|`torch`|  >= `1.2` (tested on `1.7`)|
|`torchvision`|  >= `0.4` (tested on `0.8.1`) |
|`matplotlib`|  >= `2.2.2`|
|`numpy`|  >= `1.17`|
|`tqdm`| >= `4.36.1`|
|`Pillow`| >= `9.0.0`|
|`Tensorboard`| >= `2.7.0`|


## Repository Organization

|File name         | Content |
|----------------------|------|
|`/Model`| directory containing implementation of CycleGAN model|
|`/Datasets`| directory containing implementations sketch2shoes and grayscale2rgb datasets|
|`/app.py`| skecth2shoes python application (requires pretrained weights)|
|`/colorize_image.py`| script to colorize given grayscale frame (requires pretrained weights)|
|`/train_*.py`| Training scripts for grayscale2rgb and sketch2shoes|


## Credits
* Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Neworks Jun-Yan Zhu.,- [Code](https://github.com/junyanz/CycleGAN), [Paper](https://arxiv.org/pdf/1703.10593.pdf).

