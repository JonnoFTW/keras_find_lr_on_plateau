from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from LRFinder import LRFinder


class LRFOnPlateau(ReduceLROnPlateau):
    def __init__(self, train_iterator, train_samples, batch_size,max_lr, epochs, *args, **kwargs):
        super(LRFOnPlateau, self).__init__(*args, **kwargs)
        self.train_iterator = train_iterator
        self.train_samples = train_samples
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_lr = max_lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lrf = LRFinder(self.model)
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    lrf.find_generator(self.train_iterator, self.min_lr, self.max_lr, self.epochs, self.train_samples // self.batch_size)
                    new_lr = lrf.get_best_lr_exp_weighted()
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print('\nEpoch %05d: LRFOnPlateau reducing '
                              'setting rate to %s.' % (epoch + 1, new_lr))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0
