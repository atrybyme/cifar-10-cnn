# Some Convolutional Neural Network(CNN) Architectures for Classifying CIFAR-10 Dataset

The repository contains some of the famous architectures on CNN implimented on cifar-10 dataset.

![cifar10][1]

I have used **Keras** with **Tensorflow** backend to implement all the architectures.
All the models were trained on my laptop with Nvidia Geforce GTX 960M Graphics Processor.

## Requirements:

- Python(3.5.2/2.7.14)
- Keras(2.0.8)
- tensorflow-gpu(1.3.0)

*These are what I used*

## Dataset, Training and Testing:

- To train your model from above codes download **CIFAR-10** dataset from [here][6].Extract and move all the batch files to your working directory.
- If you want to test the architectures by pretrained models over test_dataset,Download the folder containing pretrained models from [here][7].Move all the .h5 files to your working directory,make sure you have test_batch also present in the same directory.Open **testing_from_saved_model.py** and change the model_name to the name of the model you want to test and then Run it.


## Architectures and papers

*Architecture of the model in this repository is slightly different from the one proposed in the papers.This was to reduce the trainable parameters in the model and make the model easy to train.*

*No data augmentation techniques were used because the only aim was to understand and apply the models.*

- **Le-Net**
  - A classic and simple model.The model have alternate convolution and pooling layer.The output of these layer is connected to 2 Fully Connected Layer.Except the last one each layer uses *'Relu'* activation function.Last layer uses Softmax to classify.
  - [LeNet-5][2]
- **Network in Network**
  - NIN instantiate a micro neural network with a multilayer perceptron, which is a potent function approximator. The feature maps are obtained by sliding the micro networks over the input in a similar manner as CNN; they are then fed into the next layer. Deep NIN can be implemented by stacking multiple of the above described structure.
  - [Network In Network][3]
- **VGG16**
  - It is a really deep network.This network has a total of about 138 million parameters but its very easy to understand.Instead having so many hyper parameters,the model uses a much simpler network where you focus on just having conv layers that are just 3 by 3 filters with stride 1 and always use the SAME padding, and make all your max polling layers 2 by 2 with a strid of 2.
  -  [Very Deep Convolutional Networks for Large-Scale Image Recognition][4]
- **Residual Network**
  - Instead of hoping each stack of layers directly fits a desired underlying mapping, we explicitly let these layers fit a residual mapping. The original mapping is recast into F(x)+x.The residual learning framework eases the training of networks, and enables them to be substantially deeper  leading to improved performance in both visual and non-visual tasks. These residual networks are much deeper than their ‘plain’ counterparts, yet they require a similar number of parameters (weights).
  - [Deep Residual Learning for Image Recognition][5]
  
## Accuracy

|Architecture       |Training Time | Optimizer |Trainingset Accuracy|Testset Accuracy|Trainable Parameters|
|:------------------|:-------------|:----------|:-------------------|:---------------|:-------------------|
|Le-Net             |5.43 min      |Adam       |85%                 |70.4%           |62,470              |
|Network in Network |48 min        |Adam       |98.2%               |82.78%          |794,170             |
|VGG-16             |98 min        |Adam       |99.2%               |85.5%           |27,331,702          |
|Residual Network   |86.6 min      |Adam       |98.54%              |*81%*           |467,946             |

## Additional Information
Because I don't have enough machines to train the larger networks.
I decreased the epochs to 50.This decreased the accuracy.

**SGD** with **Momentum** proved to be a better optimizer(even a little) than **Adam** on the above architecture.I will soon update the optimizer.

*Also the models seems to overfit the training data. This is due to inefficient use of Dropout and due the fact that I did not used Image Augmentation. I will soon resolve this issue.*


**Please feel free to contact me for any questions and Suggestions.**


[1]: cifar10.png
[2]: http://yann.lecun.com/exdb/lenet/
[3]: https://arxiv.org/abs/1312.4400
[4]: https://arxiv.org/abs/1409.1556
[5]: https://arxiv.org/abs/1512.03385
[6]: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[7]: https://drive.google.com/drive/folders/1Y2UVn2TdkmaXZhjQbtoyPrelhkJgi9mF?usp=sharing
