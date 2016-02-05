Cifar-10 neural-net classifiers
======

The CIFAR-10 classifiers of various neural networks. Datasets from [cifar-10 datasets](http://www.cs.toronto.edu/~kriz/cifar.html).

## Requirements
[tqdm](https://github.com/noamraph/tqdm)

[chainer v1.5.1](http://chainer.org/)

[Other packages](https://github.com/pfnet/chainer#requirements) around chainer.


## Usage
### Data acquisition and shaping
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
gzip -d cifar-10-python.tar.gz
tar -xf cifar-10-python.tar
python data.py
```

### Training

Training can be done in various algorithms (optimizer = {Adam, AdaGrad, SGD}, with/without batch-normalization, dropout rate, different data augumentation methods) and neural architectures {Fully-connected neural network, several Convolutional neural networks}.  In either case, if you use GPU, add option ```-g 0``` or ```--gpu 0```.  If you want to export result figures, ```-p on``` or ```--plot on```, write log, ```-l on``` or ```--log on```, and save models ```-s on``` or ```--save on```.

#### Simple neural network
```
python  train_nn.py -d on
```
"Simple" means fully-connected. To know the model, see ```model_nn.py```
```-d on``` option makes input data normalized and augumented. To know detail, see ```datahandler.py```. The test accuracy should be around 60%.


#### Convolutional neural network

```
python  train_cnn.py -d on
```
You can change models by options ```-m alex```or ```-m alexbn```, which represents with/without batch-normalization layers. To know the model, see ```model_cnn.py```. The test accuracy should be around 85%.


#### For better accuracy...?
```
python  train_cnn_crop.py -d on
```
By cropping the images, input data is augumented further and the prediction accuracy will be even better! The test accuracy should be...


## Author

[shiba24](https://github.com/shiba24), Jan, 2016

