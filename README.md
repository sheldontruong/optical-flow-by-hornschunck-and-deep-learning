# Optical Flow Study

This repository is a study on **Optical Flow** methods, covering two main approaches:
1. **Variational Method** - Horn-Schunck algorithm
2. **Deep Learning-based Method** - FlowNet

Optical flow is the process of estimating the motion of objects between two consecutive frames in a video. This study compares traditional variational methods with modern deep learning-based approaches, highlighting their strengths, weaknesses, and performance on a short demonstration video.

## Table of Contents
- [Dataset](#dataset)
- [Methods](#methods)
  - [1. Variational Method - Horn-Schunck](#1-variational-method---horn-schunck)
  - [2. Deep Learning-based Method - FlowNet](#2-deep-learning-based-method---flownet)
- [Results](#results)
- [References](#references)

## Dataset

For this demonstration, a short video with clear motion is used to show the output of each optical flow method, which can be found in the folder `dataset`



## Methods

### 1. Variational Method - Horn-Schunck

The **Horn-Schunck** algorithm is a classic variational method for computing dense optical flow. It assumes:
- **Brightness Constancy**: Pixel intensities remain constant between frames.
- **Spatial Smoothness**: The flow field varies smoothly over the image.

The Horn-Schunck method minimizes an energy function that balances these assumptions, resulting in a smooth and dense flow field. It is effective for capturing small motions but struggles with large displacements and can be sensitive to noise.

### 2. Deep Learning-based Method - FlowNet

**FlowNet** is a deep neural network designed for estimating optical flow. It was one of the first CNN architectures developed to directly predict optical flow, making it suitable for handling larger and more complex motions.

FlowNet’s end-to-end learning approach outperforms traditional methods in scenarios involving:
- Larger displacements
- Dynamic backgrounds
- Non-rigid transformations

This project uses pretrained FlowNet models for optical flow estimation. Download the pretrained models and use them to run inference on the short video.

## Repository structure:

```
optical-flow-study/
├── data/                    # Directory for datasets
│   ├── video1/              # Place video here
│   └── ...
├── horn_schunck.py          # Implementation of Horn-Schunck algorithm for optical flow
├── flownet/                 # Directory for FlowNet implementation
│   ├── model.py             # Model definitions for FlowNet
│   ├── predict.py           # Script to run FlowNet predictions on input video using pretrained models
│   └── utils.py             # Utility functions for FlowNet
├── results/                 # Directory to store output results and visualizations
│   ├── horn_schunck/        # Results from Horn-Schunck method
│   └── flownet/             # Results from FlowNet
└── README.md                # Project documentation
```

## Results

| Method             | Strengths                                       | Weaknesses                                  |
|--------------------|-------------------------------------------------|---------------------------------------------|
| **Horn-Schunck**   | Simple, smooth flow field                       | Sensitive to noise, struggles with large motions |
| **FlowNet**        | Handles large, complex motions; end-to-end learning | Requires large datasets, computationally intensive |

A comparison of results on the short video will demonstrate the quality differences in optical flow estimation for each method.

## References

- Horn, B. K. P., & Schunck, B. G. (1981). Determining Optical Flow. *Artificial Intelligence*, 17(1-3), 185-203.
- Dosovitskiy, A., Fischer, P., Ilg, E., Hausser, P., Hazirbas, C., Golkov, V., van der Smagt, P., Cremers, D., & Brox, T. (2015). FlowNet: Learning Optical Flow with Convolutional Networks. *IEEE International Conference on Computer Vision (ICCV)*.

---

This study explores mainly how variational methods approach to solve the optical flow problem. On the other hand, a comprehensive comparison to modern methods using Deep Learning is provided to understand the limitation of traditional methods. Feel free to explore and contribute to the project!
