import numpy as np

class LinearAutoEncoder:
    # use a two layers neural network to realize the linear auto encoder
    def __init__(self, input_size, encode_size):
        self.input_size = input_size
        self.encode_szie = encode_size
        
        