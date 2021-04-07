# tensorflow-RDFSR

## Overview
This is a Tensorflow implementation for ["Image super resolution based on residual block dense connection"].
- To download the required data for training/testing, please refer to the README.MD at data directory.

## Files
- model.py : RDFSR  model definition.
- train_model.py : main training file
- test_model.py : test all the saved checkpoints
- utils.py : Data preprocessing function and image evaluation criteria

## How To Use
### Training
```shell
# if start from scratch
python train_model.py
Save the parameters to the checkpoint file every 10 training rounds
```

### Testing
```shell
# this will test all the checkpoint in ./checkpoint directory.
python test_model.py
You can modify the parameters in the checkpoint file in this file to view the picture results of training at different times
```
## Test set
```shell
This file only preprocessed the data of SET5 test set. If you need to test other data sets, 
you need to save the test set according to the Set5 file structure, and then use the
 bic_picture () function in utils to subsample it to get the low-resolution picture test_lower, 
and then up-sample the picture in test_lower to get test_highsr. Then test_lowsr, test
```

##### Results on Set 5
|  Scale    | Bicubic | RDFSR | 
|:---------:|:-------:|:----:|
| **2x** - PSNR/SSIM|   33.66/0.9929	|   37.15/0.9613	| 
| **3x** - PSNR/SSIM|   30.39/0.8682	|   32.64/0.9258	| 
| **4x** - PSNR/SSIM|   28.42/0.8104	|   31.51/0.9047	| 

##### Results on Set 14

|  Scale    | Bicubic | RDFSR |
|:---------:|:-------:|:----:|
| **2x** - PSNR/SSIM|   30.24/0.8688	|   33.22/0.9292	| 
| **3x** - PSNR/SSIM|   27.55/0.7742	|   29.17/0.8668	| 
| **4x** - PSNR/SSIM|   26.00/0.7027	|   28.87/0.8446	| 

## Requirements

Script requirements : 
- TensorFlow/Theano
- Keras
- CUDA (GPU)
- CUDNN (GPU)
- Scipy + PIL
- Numpy
- OpenCV

Tested on Windows 10, Python 3.6, TensorFlow 1.10.0, CUDA 10.1.0, CuDNN 7.5.