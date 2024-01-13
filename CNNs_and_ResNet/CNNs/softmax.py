import numpy as np
from layer import Layer

class Softmax(Layer):

    def softmax(self):
        shifted_logits =  self.input- np.max(self.input)
        tmp = np.exp(shifted_logits)
        self.output = tmp / np.sum(tmp) 
        return self.output

    def cross_entropy(self,y_true:list):
        loss = -np.sum(y_true*np.log(self.output+1e-15))
        return np.maximum(loss,0)


    def forward(self,input):
        self.input=input
        #print(self.input)
        return self.softmax()
    
    
    def backward(self, output_gradient, learning_rate):
        return self.output-output_gradient
    