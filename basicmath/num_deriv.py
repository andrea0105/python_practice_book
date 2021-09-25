import numpy as np

f_xy = lambda x : (x[0]**2 + 2*x[0])*np.log(x[1])

def numer_deriv(f, x, h=0.001, method="center"):
    if type(x) in (float, int) :
        grad = [0.0]
        x_ = [x]
        var_type = 'scalar'
    else :
        grad = np.zeros(x.shape)
        x_ = x.copy().astype('float32')
        var_type = 'vector'
        
    for i, xi in enumerate(x_) :
        original_value = x_[i]
        
        if method=='forward' :
            x_[i] = original_value + h
        else : 
            x_[i] = original_value + 0.5*h
            
        if var_type == 'scalar' :
            gradplus = f(x_[i])
        else :
            gradplus = f(x_)
            
        if method == 'forward' :
            x_[i] = original_value
        else :
            x_[i] = original_value - 0.5*h
        
        if var_type == 'scalar' :
            gradminus = f(x_[i])
        else :
            gradminus = f(x_)
            
        grad[i] = (gradplus - gradminus) / h
        
    if var_type == 'scalar' :
        return grad[0]
    else :
        return grad

res = numer_deriv(f_xy, np.array([1,2]), h=0.0001, method="center")
print(res)
