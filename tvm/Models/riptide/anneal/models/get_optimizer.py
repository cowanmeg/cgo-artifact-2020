import tensorflow as tf
import math
from .learning_schedules import *


def get_optimizer(name, lr, global_step, batch_size, num_gpus=1):
    schedules = {
        'adam': adam_piecewise,
        'sgd': sgd_piecewise,
        'cos': cosine_decay,
        'cyc': cyclic,
        'poly' : polynomial,
        'fixed' : fixed,
    }

    if 'adam' in name:
        name = 'adam'
    elif 'cos' in name:
        name = 'cos'
    elif 'cyc' in name:
        name = 'cyc'
    elif 'poly' in name:
        name = 'poly'
    elif 'fixed' in name:
        name = 'fixed'
    else:
        name = 'sgd'
    
    if math.isclose(lr, -1.0, rel_tol=1e-5):
        return schedules[name](global_step, batch_size, num_gpus)
    return schedules[name](global_step, batch_size, num_gpus, lr)
