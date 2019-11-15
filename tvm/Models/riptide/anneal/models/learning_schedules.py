import numpy as np
import tensorflow as tf
from .clr import cyclic_learning_rate

_NUM_IMAGES = 1281167


def adjust_start_lr(lr, batch_size, num_gpus, batch_denom=128):
    if num_gpus == 0:
        num_gpus = 1

    batch_size = batch_size * num_gpus

    lr_adjustment = batch_size / batch_denom
    starting_lr = lr * lr_adjustment
    return starting_lr, _NUM_IMAGES / batch_size

def fixed(global_step, batch_size, num_gpus, starting_lr=0.1):
    starting_lr, steps_per_epoch = adjust_start_lr(starting_lr,
                                                   batch_size,
                                                   num_gpus,
                                                   batch_denom=256)
    lr_schedule = starting_lr
    optimizer = tf.compat.v1.train.MomentumOptimizer(lr_schedule, 0.9)

    return optimizer, lr_schedule



def adam_piecewise(global_step, batch_size, num_gpus, starting_lr=1e-4):
    starting_lr, steps_per_epoch = adjust_start_lr(starting_lr, batch_size, num_gpus)

    lr_decay = 0.2
    lr_values = [
        starting_lr, starting_lr * lr_decay, starting_lr * lr_decay * lr_decay
    ]
    epoch_boundaries = [56, 64]
    step_boundaries = [int(x * steps_per_epoch) for x in epoch_boundaries]
    lr_schedule = tf.compat.v1.train.piecewise_constant_decay(
        global_step, boundaries=step_boundaries, values=lr_values)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_schedule,
                                                 epsilon=1e-5)
    return optimizer, lr_schedule


def sgd_piecewise(global_step, batch_size, num_gpus, starting_lr = 0.1):
    starting_lr, steps_per_epoch = adjust_start_lr(starting_lr,
                                                   batch_size,
                                                   num_gpus,
                                                   batch_denom=256)

    lr_scales = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    #lr_scales = [1, 0.5, 0.25, 0.125, 0.01, 1e-2, 1e-4]
    lr_values = [starting_lr] * len(lr_scales)
    lr_values = list(np.asarray(lr_values) * np.asarray(lr_scales))
    # epoch_boundaries = [30, 60, 85, 95]
    epoch_boundaries = [60, 80, 95, 105]
    step_boundaries = [int(x * steps_per_epoch) for x in epoch_boundaries]
    lr_schedule = tf.compat.v1.train.piecewise_constant_decay(
        global_step, boundaries=step_boundaries, values=lr_values)
    optimizer = tf.compat.v1.train.MomentumOptimizer(lr_schedule, 0.9)

    return optimizer, lr_schedule


def cosine_decay(global_step, batch_size, num_gpus, starting_lr=.0128):
    starting_lr, steps_per_epoch = adjust_start_lr(starting_lr,
                                                   batch_size,
                                                   num_gpus,
                                                   batch_denom=128)

    lr_schedule = tf.compat.v1.train.cosine_decay_restarts(
        starting_lr, global_step, 1000)

    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr_schedule,
                                                     momentum=0.9,
                                                     use_nesterov=False)

    return optimizer, lr_schedule


def cyclic(global_step, batch_size, num_gpus, starting_lr):
    max_lr, steps_per_epoch = adjust_start_lr(starting_lr,
                                              batch_size,
                                              num_gpus,
                                              batch_denom=128)

    step_size = 4 * steps_per_epoch

    lr_schedule = cyclic_learning_rate(global_step,
                                       learning_rate=1e-6,
                                       max_lr=max_lr,
                                       step_size=step_size,
                                       mode='triangular')

    # Momentum should vary inversely to learning rate.
    max_momentum = 0.9
    min_momentum = 0.0
    momentum_schedule = max_momentum - cyclic_learning_rate(
        global_step,
        learning_rate=min_momentum,
        max_lr=max_momentum,
        step_size=step_size,
        mode='triangular')

    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=lr_schedule,
        momentum=momentum_schedule,
        use_nesterov=False)
    return optimizer, lr_schedule


def polynomial(global_step, batch_size, num_gpus, starting_lr=0.1):

    base_lr = starting_lr
    end_learning_rate = 1e-4 # darknet doesn't have end_learning_rate
    power = 4.0
    max_batches = 1e6 #800000
    burnin_batches = 1000


    decay_steps = max_batches
    num_warmup_steps = burnin_batches
    initial_learning_rate = base_lr * (num_gpus * float(batch_size) / 128.0) 

    lr = tf.compat.v1.train.polynomial_decay(
                initial_learning_rate,
                global_step,
                decay_steps,
                end_learning_rate=end_learning_rate,
                power=power)

    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_learning_rate = initial_learning_rate * (global_steps_float / warmup_steps_float) ** power

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        lr = (1.0 - is_warmup) * lr + is_warmup * warmup_learning_rate

    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=lr,
                momentum=0.9,
                use_nesterov=False)
    
    return optimizer, lr



