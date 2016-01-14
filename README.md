# cifar-10
The CIFAR-10 classifier example.

## USAGE
Data acquisition and shaping:
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
gzip -d cifar-10-python.tar.gz
tar -xf cifar-10-python.tar
python data_cifar.py
```

Training: Simple neural network (using net.py)
```
python  train_cifar_simpleNN.py
```

Training: Convolutional neural network (using alex.py, googlenet.py, etc...)

```
python  train_cifar_CNN.py
```
