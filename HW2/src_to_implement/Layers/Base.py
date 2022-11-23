class BaseLayer(object):
    def __init__(self):
        self.trainable = False
        self.weights = []
    
    def forward(self):
        raise NotImplementedError
    
    def __call__(self, *args):
        return self.forward(*args)