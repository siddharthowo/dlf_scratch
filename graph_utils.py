class Value:
    def __init__(self,data,_children=(),_op="",):
        assert isinstance(data, (int, float)), f"Data must be a number, received {type(data).__name__}"
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._backward=lambda: None
        self._op=_op

    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        res=Value(self.data+other.data,(self,other),_op="+")

        def _backward():
            self.grad+=1.0*res.grad
            other.grad+=1.0*res.grad

        res._backward=_backward
        return res

    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        res=Value(self.data*other.data,(self,other),_op="*")

        def _backward():
            self.grad+=other.data*res.grad
            other.grad+=self.data*res.grad

        res._backward=_backward
        return res

    def __radd__(self,other):
        return self+other

    def __rmul__(self,other):
        return self*other

    def __neg__(self):
        return -1*self

    def __sub(self,other):
        return self + (-other)

    def relu(self):
        res=Value(0.0 if self.data<0 else self.data,(self,),"ReLU")

        def _backward():
            self.grad+=(0.0 if self.data<0 else 1.0)*res.grad

        res._backward=_backward
        return res

    def backward(self):
        topo=[]
        visited=set()

        def build_topo(v):
            if(v not in visited):
                visited.add(v)
                for p in v._prev:
                    build_topo(p)
                topo.append(v)

        build_topo(self)

        self.grad=1
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"