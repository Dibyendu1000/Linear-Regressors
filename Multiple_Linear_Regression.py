import numpy as np

def coef(X,y):
    X_T=X.transpose()
    B=np.matmul(X_T,X)
    B_in=np.linalg.inv(B)
    B=np.matmul(np.matmul(B_in,X_T),y)
    return B

def main(X,y):
    n=int(input("Enter number of rows:"))
    for i in range(n):
        row=list(map(float,input('Enter features seperated by comma:').split(',')))
        X.append(row)
    arr=fit(X,y)
    X=np.array(X)
    resp=list(map(float,input("Enter values of response:").split(',')))
    y=np.array(resp)
    b=coef(arr[0],arr[1])
    test_case=list(map(float,input("Enter test case:").split(',')))
    
    y_1=np.matmul(test_case,b)
    print(y_1)

main()
