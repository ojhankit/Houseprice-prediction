import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess
from architecture import ANN

if __name__ == '__main__':
    
    df = preprocess()
    
    X ,y = df.iloc[:,1:] ,df.iloc[:,0:1]
    
    X_train ,X_test ,y_train ,y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42)
    
    architecture = [X_train.shape[1],64,32,1]
    ann = ANN(arch=architecture)
    ann.fit(X_train,y_train)
    
 