# CNN vs. ViT on Elephants

CNN vs. ViT on Elephants - Image Binary Classification on African vs. Asian Elephants Dataset

<h1 align="center">
  <br>
CNN vs. ViT on Elephants <br> Image Binary Classification on African vs. Asian Elephants Dataset
  <br>
</h1>
  <h3 align="center">
    <a href="https://github.com/ArielLulinsky">Ariel Lulinsky</a> •
    <a href="https://hadar-hai.github.io/">Hadar Hai</a>

  </h3>
<h3 align="center">Deep Learning Course Final Project</h3>

<h4 align="center">046211 ECE Technion 2024</h4>

<h4 align="center"><a href="להשלים">Project Presentation Video</a>

<h4 align="center">
    <a href="להשלים"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</h4>

<p align="center">
<img src="https://github.com/hadar-hai/vit-vs-cnn-on-elephants/assets/64587231/b365f642-bf68-43ac-88f2-49fab6aae3ee"/>
</p>

## Prerequisites

| Library           | Version    | Why                                                |
|-------------------|------------|-------------------------------------------------------|
| `matplotlib`      | -          | Plotting and visualization                            |
| `time`            | -          | Time-related functions                                |
| `os`              | -          | Operating system interface                            |
| `copy`            | -          | Shallow and deep copy operations                      |
| `pdb`             | -          | Python Debugger                                       |
| `pandas`          | -          | Data manipulation and analysis                        |
| `random`          | -          | Random number generation                              |
| `shutil`          | -          | High-level file operations                            |
| `torch`           | >= 1.0.0   | Deep learning framework                               |
| `torch.nn`        | -          | Neural network operations                             |
| `torch.nn.functional` | -      | Functional interface for neural network operations    |
| `torch.utils.data` | -         | Utilities for data loading and processing             |
| `torchvision.datasets` | -     | Datasets and transformations for vision tasks         |
| `torchvision.transforms` | -   | Image transformations for vision tasks                |
| `kornia`          | -          | Differentiable computer vision library for PyTorch    |
| `kornia.augmentation` | -      | Image augmentation                                    |
| `kornia.geometry.transform` | - | Geometric transformations for images                  |
| `PIL`             | -          | Python Imaging Library for image processing           |
| `tkinter`         | -          | GUI Toolkit for Python                                |
| `datetime`        | -          | Date and time manipulation                            |
| `ImageTk`         | -          | PIL extension for displaying images in Tkinter        |
| `Image`           | -          | PIL module for image processing                       |


## Datasets
| Dataset           | Notes                         | Link                                                                                |
|-------------------|------------------------------------|--------------------------------------------------------------------------------------|
| Asian vs. African Elephant Image Classification  | The dataset is not available in this repository, please download it from the link | [Kaggle](https://www.kaggle.com/datasets/vivmankar/asian-vs-african-elephant-image-classification) |

## Repository Organization

| File name                                            | Content                                                                                     |
|------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `/checkpoints (now models)`                                       | directory for trained checkpoints       and models                                                |
| `/assets`                                            | directory for assets (gifs, images, etc.)                                  |
| `/docs`                                              | various documentation files                                                                 |
| `/notebooks`                                         | Jupyter Notebooks used for training and evaluation                                         |
| `/histories`                                         | models checkpoints training history by epochs                                   |
| `/logs`                                         | human classification results per person                                   |
| `human_classification.py`                                 | `tkinter`-based interactive GUI to collect data on human classification performance on the African vs. Asian dataset                |
| `environment.yml`                                    | Anaconda environment file to install the required dependencies                              |
| `requirements.txt`                                   | requirements file for `pip`                                                                 |

