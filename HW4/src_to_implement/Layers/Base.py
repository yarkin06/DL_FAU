class BaseLayer(object):
    def __init__(self):
        self.trainable = False
        self.weights = []
        self.testing_phase = False