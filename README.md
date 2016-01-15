# cifar-10
The CIFAR-10 classifier example. Datasets from [cifar-10 datasets](http://www.cs.toronto.edu/~kriz/cifar.html).

## USAGE
Data acquisition and shaping:
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
gzip -d cifar-10-python.tar.gz
tar -xf cifar-10-python.tar
python data.py
```

Training: Simple neural network (using net.py)
```
python  train_nn.py
```

Training: Convolutional neural network (using cnn_alex.py, cnn_googlenet.py, etc...)

```
python  train_cnn.py
```

Both, if you use GPU, option ```-g 0```. If you want to export result figures, ```-p on``` and write log, ```-l on```.
