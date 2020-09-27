import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque


def get_config():
    iterations = 1e4
    n_max_steps = 200
    n_fixed_steps = 2048
    discount_factor = 0.95
    

class ActorCritic(keras.Model):
    def __init__(self) :
        ac_layer1 = keras.layers.Dense()