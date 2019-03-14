import numpy as np 
import matplotlib.pyplot as plt

def estimate_coef(x,y):
    #number of observations/points
    n=np.size(x)

    #mean of x and y vector
    m_x, m_y =np.mean(x),np.mean(y)

    #calculating cross-deviation and devaiation about x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x

    #calculating regression coefficients
    b_1= SS_xy/SS_xx
    b_0=m_y -b_1*m_x

    return(b_0,b_1)

def plot_regression_line(x,y,b):
    #plotting the actual points as scatter plot
    plt.scatter(x,y,color="m",marker="o",s=30)

    #predicted response vectore
    y_pred=b[0]+b[1]*x

    #plotting the regression line
    plt.plot(x,y_pred,color="g")

    #putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    #function to show plot
    plt.show()
    
def main():
    #training set
    x=list(map(float,input("Enter training feature set seperated by comma:").split(',')))
    y=list(map(float,input("Enter training response set seperated by comma:").split(',')))
    x=np.array(x)
    y=np.array(y)

    #estimating coefficients
    b=estimate_coef(x,y)
    print(f"Estimated coefficients:\nb_0={b[0]}\nb_1={b[1]}")

    #plotting regression line
    plot_regression_line(x,y,b)
main()
