# ViT vs. CNN on Elephants

ViT vs. CNN on Elephants - Image Binary Classification on African vs. Asian Elephants Dataset

<h1 align="center">
  <br>
ViT vs. CNN on Elephants <br> Image Binary Classification on African vs. Asian Elephants Dataset
  <br>
</h1>
  <h3 align="center">
    <a href="https://github.com/ArielLulinsky">Ariel Lulinsky</a> •
    <a href="https://hadar-hai.github.io/">Hadar Hai</a>

  </h3>
<h3 align="center">Deep Learning Course Final Project</h3>

<h4 align="center">046211 ECE Technion 2024</h4>

<h4 align="center"><a href="להשלים">Project Presentation Video</a>

# Elephant Image Classification: ViT vs CNN

This project focuses on evaluating Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for image classification tasks, specifically distinguishing between Asian elephants and African elephants. The project focuses on the models of MobileNet, ResNet, ViT-b-16 and DINOv2. Leveraging transfer learning with pre-trained models, we aim to achieve accurate analysis and classification of images depicting these majestic creatures. By utilizing publicly available datasets, this project contributes to elephant conservation and research efforts.

### Key Findings

- **ViT Models**: Outperformed the CNN models, with self-supervised ViT models achieving the highest accuracy, highlighting their effectiveness in capturing spatial relationships.
- **Importance of Pretrained Dataset Size**: Achieving good results for ViT on small dataset requires utilizing pretrained model on a huge dataset such as ImageNet. 

## Results

The following images depict the performance metrics and visualizations from the experiments conducted in this project for best CNN model (ResNet) and best ViT model (DINOv2):

<p align="center">
    <img src="https://github.com/hadar-hai/vit-vs-cnn-on-elephants/assets/64587231/34e1f1aa-2a45-4128-9ae5-bf8475d58fcc" width="400">
</p>
<p align="center">
    <img src="https://github.com/hadar-hai/vit-vs-cnn-on-elephants/assets/64587231/c6f59665-51ce-49e2-93a8-b5696cd1ce50" width="500">
</p>

#### Image Human Classification Tool

<p align="center">
  <img src="https://github.com/hadar-hai/vit-vs-cnn-on-elephants/blob/main/assets/human_classification_demo.gif" alt="Human Classification Demo">
</p>

#### CNN Results - Activation Map
<p align="center">
    <img src="https://github.com/hadar-hai/vit-vs-cnn-on-elephants/assets/64587231/f89b254f-2a87-48c1-a04e-0587d97ba51a" width="500">
</p>

## Conclusion

Overall, the project provides valuable insights into the performance of different models and underscores the importance of considering factors such as model architecture, dataset size and pretraining strategies in image classification tasks. Specifically, when dealing with small datasets, Self-supervised ViT (pretrained on large dataset) outperforms other methods.


## Prerequisites

| Library                  | Why                                                             |
|--------------------------|-----------------------------------------------------------------|
| `matplotlib`             | Plotting and visualization                                      |
| `time`                   | Time-related functions                                          |
| `os`                     | Operating system interface                                      |
| `copy`                   | Shallow and deep copy operations                                |
| `PIL`                    | Python Imaging Library for image processing                     |
| `cv2`                    | OpenCV library for computer vision tasks                        |
| `pandas`                 | Data manipulation and analysis                                  |
| `torch`                  | Deep learning framework                                         |
| `torchvision`            | Datasets and transformations for vision tasks                   |
| `sklearn`                | Machine learning library                                        |
| `IPython`                | Displaying images in IPython                                    |
| `kornia`                 | Differentiable computer vision library for PyTorch              |
| `pytorch_grad_cam`       | Package for visualizing convolutional neural network activation maps |
| `tkinter`                | GUI Toolkit for Python                                          |
| `datetime`               | Date and time manipulation                                      |
| `random`                 | Random number generation                                        |



## Datasets
| Dataset           | Notes                         | Link                                                                                |
|-------------------|------------------------------------|--------------------------------------------------------------------------------------|
| Asian vs. African Elephant Image Classification  | The dataset is not available in this repository, please download it from the link | [Kaggle](https://www.kaggle.com/datasets/vivmankar/asian-vs-african-elephant-image-classification) |

## Repository Organization

| File name                                            | Content                                                                                     |
|------------------------------------------------------|---------------------------------------------------------------------------------------------|
| `/checkpoints`                                       | directory for trained checkpoints models and histories                                                |
| `/assets`                                            | directory for assets (gifs, images, etc.)                                  |
| `/docs`                                              | various documentation files                                                                 |
| `/notebooks`                                         | Jupyter Notebooks used for training and evaluation                                         |
| `/logs`                                         | human classification results per person                                   |
| `/tools/human_classification.py`                                 | `tkinter`-based interactive GUI to collect data on human classification performance on the African vs. Asian dataset                |
| `requirements.txt`                                   | requirements file for `pip`                                                                 |

## References

- Jacob Gil. "pytorch-grad-cam repository." GitHub. [https://github.com/jacobgil/pytorch-grad-cam/tree/master](https://github.com/jacobgil/pytorch-grad-cam/tree/master)

- Luthei. "VIT vs CNN on WikiArt." Kaggle. [https://www.kaggle.com/code/luthei/vit-vs-cnn-on-wikiart](https://www.kaggle.com/code/luthei/vit-vs-cnn-on-wikiart)

- Vivmankar. "Asian vs African Elephant Image Classification Dataset." Kaggle. [https://www.kaggle.com/datasets/vivmankar/asian-vs-african-elephant-image-classification](https://www.kaggle.com/datasets/vivmankar/asian-vs-african-elephant-image-classification)

- Nasruddin Az. "Elephclass: Asian vs African Elephants Classifier." Kaggle. [https://www.kaggle.com/code/nasruddinaz/elephclass-asian-vs-african-elephants-classifier](https://www.kaggle.com/code/nasruddinaz/elephclass-asian-vs-african-elephants-classifier)

- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, et al.
  [ArXiv](https://arxiv.org/abs/2010.11929v2)


## Acknowledgments

This project is a part of the ECE 046211 Deep Learning course at the Technion. We would like to express our gratitude to [Tal Daniel](https://taldatech.github.io/) and [Prof. Daniel Soudry](https://soudry.github.io/) for their guidance and support throughout this project and the course.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

