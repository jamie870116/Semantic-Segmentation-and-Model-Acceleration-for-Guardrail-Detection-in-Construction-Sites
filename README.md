# Semantic-Segmentation-and-Model-Acceleration-for-Guardrail-Detection-in-Construction-Sites
This study aims to adopt different models and apply various model compression and acceleration techniques to improve the performance of segmentation and computing efficiency of the semantic segmentation models for small objects such as safety guardrails. Here, the object is to help the monitoring personnel to judge whether the safety guardrails are set correctly, and reduce the accident rate in construction sites. Due to the scarcity of data, caused by the restrictions of construction site safety regulations and the difficulty of data labeling, this study adopts the method of data augmentation to assist the training process of the model. In addition, in response to the model with hardware performance and a large amount of model parameters, it is found that using input images of different sizes for different models can ensure its segmentation performance and successfully perform guardrail identification according to the experiments. As a result, all models achieve above 0.54 in IoU. In this study, Ghost Module is chosen as the acceleration method, and experiments have confirmed that this acceleration method can help improve the computing efficiency, and make the performance of segmentation of the model up to an IoU of 0.65. Although running on edge devices cannot achieve the level of real-time segmentation, after model acceleration, the time required for an image is still significantly decreased by more than 110 percent. Also, since the guardrail is a static object, there is no need for the fast identification frequency. Finally, in order to further reduce the computational complexity of the model, this study uses model pruning to compress the overall model size. According to the results of the experiments, it is found that there is indeed a problem of redundant weights in the model. After discarding a certain degree of redundant weights by the L1 norm and adopting fine-tuning, it can effectively improve the model's ability to segment guardrails. 

# Usage
## Dataset

### **Source**
The dataset used in this project is sourced from the "Application and Promotion of Artificial Intelligence in Hazard Identification of Construction Engineering" project of the Occupational Safety and Health Administration, Ministry of Labor in 2021. As a result, the dataset is not publicly available.

### **Pre-processing**
Including data labeling, data augmentation and resizing.

**The following packages are required for Pre-processing**
- cv2
- matplotlib
- scikit-learn
- numpy

## Experiments
**The following packages are required for Pre-processing**
- torch
- torch.nn
- torch.utils
- torch.utils
- torch.optim
- torchvision
- torchsummary
- matplotlib.pyplot
- numpy
- PIL
- random
- cv2
- sklearn
- re
- math


### Model
**The following models are implemented**
- EDANet
- Unet++
- DeepLab V3+

### Model Acceleration
Ghost Module and Model Pruning are used in this project


# Credits
This project was created by [Jamie/Chieh-Ying](https://github.com/jamie870116).

# Others
This repository is the code of the thesis ["用於營建工地安全護欄偵測之語義分割與模型加速"](https://hdl.handle.net/11296/7d56rx) writen by [Chieh-Ying Lai](https://github.com/jamie870116), and instructed by [Meng-Hsiun Tsai](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/ccd=ovPcND/search?q=ade=%22Meng-Hsiun%20Tsai%22.&searchmode=basic#result).

Please feel free to contect [me](https://github.com/jamie870116) for any further questions.
