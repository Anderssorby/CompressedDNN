import abc
import time
import warnings
from collections import deque

import numpy as np

from odin.utils.progbar import Progbar


class Callback(abc.ABC):
    """
    A general callback base class
    """

    params: dict

    def __init__(self):
        self.params = {}
        self.model_wrapper = None

    def __call__(self, event="", *args, **kwargs):
        if hasattr(self, event):
            return getattr(self, event)(*args, **kwargs)

    def set_params(self, params):
        self.params = params

    def set_model_wrapper(self, model_wrapper):
        self.model_wrapper = model_wrapper


class TrainingCallback(Callback, abc.ABC):

    def __init__(self):
        super(TrainingCallback, self).__init__()
        self.validation_data = None
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class GANSampler(TrainingCallback):

    def __init__(self, period=1, batch_size=100):
        self.batch_size = batch_size
        self.period = period
        self.epochs_since_last_sample = 0
        super(GANSampler, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_sample += 1
        if self.epochs_since_last_sample >= self.period:
            self.epochs_since_last_sample = 0

            dataset = self.model_wrapper.dataset
            x_target, y_input_condition = next(
                dataset.random_batch_generator(self.batch_size, data_type="train"))
            self.model_wrapper.plot_generated_batch(x_target, y_input_condition, "training",
                                                    epoch)
            x_full_batch, x_sketch_batch = next(
                dataset.random_batch_generator(self.batch_size, data_type="test"))
            self.model_wrapper.plot_generated_batch(x_full_batch, x_sketch_batch, "validation",
                                                    epoch)


class ModelCheckpoint(TrainingCallback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.model_wrapper:
                            self.model_wrapper.save(epoch=epoch, checkpoint=True, **logs)
                        else:
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.model_wrapper:
                    self.model_wrapper.save(epoch=epoch, is_checkpoint=True, **logs)
                else:
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)


class ProgbarLogger(TrainingCallback):
    """Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).

    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode='samples',
                 stateful_metrics=None):
        super(ProgbarLogger, self).__init__()
        self.progbar = None
        self.target = None
        self.verbose = None
        self.seen = None
        self.epochs = None
        self.log_values = None

        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
            if self.use_steps:
                target = self.params['steps']
            else:
                target = self.params['samples']
            self.target = target
            self.progbar = Progbar(target=self.target,
                                   verbose=self.verbose,
                                   stateful_metrics=self.stateful_metrics)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values)


class CallbackManager(object):
    callbacks: list
    model = None
    params: dict = {}

    def __init__(self, callbacks, queue_length=10):
        self.callbacks = callbacks
        self.queue_length = queue_length
        self.model_wrapper = None

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def set_model_wrapper(self, model_wrapper):
        self.model_wrapper = model_wrapper
        for callback in self.callbacks:
            if isinstance(callback, Callback):
                callback.set_model_wrapper(model_wrapper)

    def set_params(self, params: dict):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def add_callbacks(self, callbacks: list):
        self.callbacks += callbacks
        for callback in callbacks:
            callback.set_model(self.model)
            callback.set_model_wrapper(self.model_wrapper)
            callback.set_params(self.params)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0. and
                delta_t_median > 0.95 * self._delta_t_batch and
                delta_t_median > 0.1):
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0. and
                (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.'
                          % delta_t_median)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)

    def __call__(self, event="", *args, **kwargs):

        if hasattr(self, event):
            return getattr(self, event)(*args, **kwargs)
