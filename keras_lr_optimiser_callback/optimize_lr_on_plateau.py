from tensorflow.python.keras.callback import Callback
from .lr_finder import LRFinder

class OptimizeLROnPlateau(Callback):
    """Find the best learning rate when a metric has stopped improving.
    Models often benefit from changing the learning rate when learning stagnates
    This callback monitors a valu  and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is optimised.
    # Example
    ```python
    reduce_lr = OptimizeLROnPlateau(monitor='val_loss', op
                                  patience=5, max_lr=1,min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        max_lr: maximum learning rate
        sma: number of batches over which to smooth loss gradients when
        selecting a new lr
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, monitor='val_loss', patience=10,
                 verbose=0, mode='auto', cooldown=0,max_lr=1, min_lr=0.0001,
                 **kwargs):
        super(OptimizeLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('OptimizeLROnPlateau '
                             'does not support a factor >= 1.0.')

        self.max_lr = max_lr
        self.min_lr = min_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Optimize LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    lrf = LRFinder(self.model)
                    lrf.find_generator(generator, self.start_lr, self.max_lr, epochs=1, steps_per_epoch=self.steps_per_epoch)
                    new_lr = lrf.get_best_lr(self.sma, n_skip_beginning=self.n_skip_beginning, n_skip_end=self.n_skip_end):
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print('\nEpoch %05d: OptimizeLROnPlateau reducing '
                              'learning rate to %s.' % (epoch + 1, new_lr))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0
