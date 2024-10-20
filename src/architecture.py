import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

class ANN:
    def __init__(self,arch:list)->None:
        
        self.weights = {}
        self.bias = {}
        self.layers = len(arch)-1
        
        # Initialize weights
        for i in range(len(arch)-1):
            
            x,y = arch[i],arch[i+1]
            self.weights[i+1] = np.random.randn(x,y) * 0.01
            self.bias[i+1] = np.zeros(shape=(y))
        
        print("Weights:")
        for key, value in self.weights.items():
            print(f"Layer {key}: \n{value}\n")

        print("Biases:")
        for key, value in self.bias.items():
            print(f"Layer {key}: \n{value}\n")    
    
    def forward(self,X,layer_num):
        result = np.dot(X,self.weights[layer_num]) + self.bias[layer_num]
        return sigmoid(result)

    
    def backward(self,X,y,lr=0.01):
        n_s = X.shape[0]
        A = {}
        A[0] = X
        
        for i in range(1,self.layers+1):
            A[i] = self.forward(A[i-1] ,i)
        
        dZ = A[self.layers] - y.reshape(-1,1)
        
        for i in reversed(range(1 ,self.layers+1)):
            dW = np.dot(A[i-1].T, dZ) / n_s  # Gradient of weights
            dB = np.sum(dZ, axis=0) / n_s    # Gradient of bias
            dZ = np.dot(dZ, self.weights[i].T) * sigmoid_derivative(A[i-1])  # Backpropagate the error
            
            # Update weights and biases
            self.weights[i] -= lr * dW
            self.bias[i] -= lr * dB
    
    def fit(self, X_train, y_train, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.backward(X_train, y_train, learning_rate)
            if epoch % 100 == 0:
                predictions = self.predict(X_train)
                loss = np.mean((predictions - y_train)**2)  # Mean squared error loss
                print(f'Epoch {epoch}, Loss: {loss}')
                
        print("Training Complete")
        print("Weights:")
        for key, value in self.weights.items():
            print(f"Layer {key}: \n{value}\n")

        print("Biases:")
        for key, value in self.bias.items():
            print(f"Layer {key}: \n{value}\n")   
        
    
    def predict(self, X):
        input = X
        for i in range(1, self.layers+1):
            input = self.forward(input, i)
        return input