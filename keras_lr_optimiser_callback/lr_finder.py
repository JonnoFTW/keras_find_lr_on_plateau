from matplotlib import pyplot as plt
import math
from keras.callbacks import LambdaCallback
import keras.backend as K
import numpy as np


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = math.inf
        self.tmp_fname = 'lrfinder_tmp.h5'

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        # if math.isnan(loss) or loss > self.best_loss * 4:
        #     self.model.stop_training = True
        #     return

        self.best_loss = min(loss, self.best_loss)

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def _reset(self):
        self.lrs.clear()
        self.losses.clear()
        self.best_loss = math.inf

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):
        self._reset()
        num_batches = epochs * x_train.shape[0] / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))

        # Save weights into a file
        self.model.save_weights(self.tmp_fname)

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch,
                                                      logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=[callback])

        # Restore the weights to the state before model fitting
        self.model.load_weights(self.tmp_fname)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
        self._reset()
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(generator)
            except (ValueError, NotImplementedError) as e:
                raise e('`steps_per_epoch=None` is only valid for a'
                        ' generator based on the '
                        '`keras.utils.Sequence`'
                        ' class. Please specify `steps_per_epoch` '
                        'or use the `keras.utils.Sequence` class.')
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (1. / float(steps_per_epoch * epochs))

        # Save weights into a file
        self.model.save_weights(self.tmp_fname)

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=self.on_batch_end)

        self.model.fit_generator(generator=generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[callback],
                                 **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.load_weights(self.tmp_fname)
        # os.unlink(self.tmp_fname)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel(r"Rate of Loss Change $\frac{dl}{dlr}$")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(lrs, derivatives, label="Loss Change")
        plt.xscale('log')
        plt.ylim(y_lim)

    def plot_exp_loss(self, beta=0.98, n_skip_beginning=10, n_skip_end=5):
        exp_loss = self.exp_weighted_losses(beta)[n_skip_beginning:-n_skip_end]
        exp_der = self.exp_weighted_derivatives(beta)[n_skip_beginning:-n_skip_end]
        best_lr_idx = min(enumerate(zip(exp_der[n_skip_beginning:-n_skip_end], self.lrs[n_skip_beginning:-n_skip_end])),
                          key=lambda x: x[1])[0]

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], exp_loss, label="Loss", markevery=[best_lr_idx])
        plt.ylabel("Exponentially Weighted Loss")
        plt.xlabel("Learning Rate (log scale)")

        plt.plot()
        plt.xscale('log')

    def plot_exp_loss_change(self, beta=0.98, n_skip_beginning=10, n_skip_end=5):
        exp_der = self.exp_weighted_derivatives(beta)[n_skip_beginning:-n_skip_end]
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], exp_der, label=r"exp weighted loss change")
        plt.ylabel(r"Exponentially Weighted Loss Change $\frac{dl}{dlr}$")
        plt.xlabel("Learning Rate (log scale)")
        plt.xscale('log')

    def get_best_lr_exp_weighted(self, beta=0.98, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.exp_weighted_derivatives(beta)
        return min(zip(derivatives[n_skip_beginning:-n_skip_end], self.lrs[n_skip_beginning:-n_skip_end]))[1]

    def exp_weighted_losses(self, beta=0.98):
        losses = []
        avg_loss = 0.
        for batch_num, loss in enumerate(self.losses):
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            losses.append(smoothed_loss)
        return losses

    def exp_weighted_derivatives(self, beta=0.98):
        derivatives = [0]
        losses = self.exp_weighted_losses(beta)
        for i in range(1, len(losses)):
            derivatives.append((losses[i] - losses[i - 1]) / 1)
        return derivatives

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        return min(zip(derivatives[n_skip_beginning:-n_skip_end], self.lrs[n_skip_beginning:-n_skip_end]))[1]
