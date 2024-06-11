#rewrite the backprop from ipynb 

class Tensor:
    
    def __init__(self,data,children=()):
        
        self.data = data
        self._prev = set(children)
        self.grad = 0.0
        self._backward = lambda : None
        
        
        
        pass