import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')



def plot_result(train_ac, test_ac, train_mean_loss, test_mean_loss, savename='result.jpg'):
    fig = matplotlib.pyplot.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ep = np.arange(len(train_ac)) + 1

    ax1.plot(ep, train_ac, color="blue", linewidth=2.5, linestyle="-", label="Train")
    ax1.plot(ep, test_ac, color="red",  linewidth=2.5, linestyle="-", label="Test")

    ax2.plot(ep, train_mean_loss, color="blue", linewidth=2.5, linestyle="-", label="Train")
    ax2.plot(ep, test_mean_loss, color="red",  linewidth=2.5, linestyle="-", label="Test")

    ax1.set_title("Accuracy")
    ax2.set_title("Mean Loss")

    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")
    fig.tight_layout()
    matplotlib.pyplot.legend(loc='upper right')

    matplotlib.pyplot.savefig(savename)

