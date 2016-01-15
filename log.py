
import logging


def write_nn(N, N_test, n_inputs, n_units, n_outputs, batchsize, netstructure, stime, etime,
             train_mean_loss, train_ac, test_mean_loss, test_ac, epoch, LOG_FILENAME='log.txt'):
    logging.basicConfig(filename=LOG_FILENAME,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s'
                        )
    logging.info(
        'New trial **************************************************\n'
        'All data: %d frames, train: %d frames / test: %d frames.\n'
        '   Inputs = %d, Units= %d, Outputs = %d, Batchsize = %d.\n'
        '   Network = %s'
        '   Total Time = %.3f sec.\n'
        '   Epoch: 1,  train mean loss=  %.5f, accuracy=  %.5f\n'
        '              test mean loss=  %.5f, accuracy=  %.5f\n'
        '   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n'
        '              test mean loss=  %.3f, accuracy=  %.3f\n',
        N + N_test, N, N_test,
        n_inputs, n_units, n_outputs, batchsize,
        netstructure,
        etime-stime,
        train_mean_loss[0], train_ac[0],
        test_mean_loss[0], test_ac[0],
        epoch, train_mean_loss[-1], train_ac[-1],
        test_mean_loss[-1], test_ac[-1]
        )
    f = open(LOG_FILENAME, 'rt')
    try:
        body = f.read()
    finally:
        f.close()
    print('FILE:')
    print(body)


def write_cnn(N, N_test, batchsize, netstructure, stime, etime,
              train_mean_loss, train_ac, test_mean_loss, test_ac, epoch, LOG_FILENAME='log.txt'):
    logging.basicConfig(filename=LOG_FILENAME,
                        level=logging.DEBUG,
                        format='%(asctime)s %(message)s'
                        )
    logging.info(
        'New trial **************************************************\n'
        'All data: %d frames, train: %d frames / test: %d frames.\n'
        '   Network = %s, Batchsize = %d.\n'
        '   Total Time = %.3f sec.\n'
        '   Epoch: 1,  train mean loss=  %.5f, accuracy=  %.5f\n'
        '              test mean loss=  %.5f, accuracy=  %.5f\n'
        '   Epoch: %d, train mean loss=  %.5f, accuracy=  %.5f\n'
        '              test mean loss=  %.3f, accuracy=  %.3f\n',
        N + N_test, N, N_test,
        netstructure, batchsize,
        etime-stime,
        train_mean_loss[0], train_ac[0],
        test_mean_loss[0], test_ac[0],
        epoch, train_mean_loss[-1], train_ac[-1],
        test_mean_loss[-1], test_ac[-1]
        )
    f = open(LOG_FILENAME, 'rt')
    try:
        body = f.read()
    finally:
        f.close()
    print('FILE:')
    print(body)
