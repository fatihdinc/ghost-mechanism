import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

deltat = .1;
eps = 0
T = 100;
x0 = 0;

xstar = 10
r_star = np.pi**2/(4*(T)**2);

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x,0)

def forward_network(x,r,deltat = 1,eps = 0):
    return x + deltat*x**2 + deltat*r +  np.sqrt(deltat)* eps *np.random.normal(0,1)

def output_network(x):
    return sigmoid(a*(x-xstar))+0.


def err_func(r,T):
    t_star = np.pi/(2*np.sqrt(r))
    t_star[r<=0]=0
    temp = np.abs(T-t_star ) *(r>np.pi**2/(16*T**2)) + T *(r<=np.pi**2/(16*T**2))
    return temp



a_all = [0.01,0.1,0.5,1,2]


r_vals = np.linspace(-1*r_star,2*r_star,100)
n_time = round(2*T/deltat)

dr = r_vals[1] - r_vals[0]

err_gt = err_func(r_vals,T);

plt.plot(r_vals, abs(np.gradient(err_gt, dr))/max(abs(np.gradient(err_gt, dr))),color = 'black')
plt.yscale('log')

cmap = plt.cm.Reds
colors = cmap(np.linspace(0.1, 0.9, len(a_all)))

for l,a in enumerate(a_all):
    error_all = np.zeros(r_vals.shape[0])+np.nan
    for k in range(r_vals.shape[0]):
        x= np.zeros(n_time+1) + np.nan
        x[0] = x0;
        r = r_vals[k]
        error = 0;
        for i in range(n_time):
            x[i+1] = forward_network(x[i],r,deltat,eps)
            if x[i+1]>1e6:
                x[i+1] = 1e6
            output = output_network(x[i+1])
            error = error+ (output - ((i+1-T/deltat)>0 ) )**2 
            
        error_all[k] = error*deltat;
    
    plt.plot(r_vals, abs(np.gradient(error_all, dr))/max(abs(np.gradient(error_all, dr))),label = a,color = colors[l])


#plt.axvline(r_star,color = 'orange',ls = '--')
plt.legend()
plt.savefig('gradient.pdf')
plt.show()


