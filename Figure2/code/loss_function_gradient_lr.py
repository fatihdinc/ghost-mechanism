import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from tqdm import tqdm


def err_func(r,T):
    if r<=0:
        return 1
    else:
        t_star = np.pi/(2*np.sqrt(r))
        
    temp = np.abs(T-t_star ) *(r>np.pi**2/(16*T**2)) + T *(r<np.pi**2/(16*T**2))    
    return temp/T


def gradient_loss(r, T):
    # Compute the three key values for comparison
    value1 = np.pi**2 / (16 * T**2)
    value2 = np.pi**2 / (4 * T**2)
    
    if value1 <= r <= value2:
        return -np.pi / (4 * r**(3/2))
    elif r > value2:
        return np.pi / (4 * r**(3/2))
    else:  # r < value1
        return 0


def run_experiment(T,alph):
    
    r_opt = np.pi**2/(4*T**2)
    
    r_cur = np.zeros([num_epoch+1])
    cur_los = np.zeros([num_epoch+1])
    r_cur[0] = 10*r_opt + np.random.normal(0,r_opt/10)
    cur_los[0] = err_func(r_cur[0],T);
    for i in range(num_epoch):
        cur_grad = gradient_loss(r_cur[i], T)
        r_cur[i+1] = r_cur[i] - alph * cur_grad
        cur_los[i+1] = err_func(r_cur[i+1],T);
    return r_cur, cur_los

T = 100;
num_epoch = 10000
r_opt = np.pi**2/(4*T**2)
num_exp = 100;
alph_all = np.logspace(-11,-8,num_exp)
cur_los = np.zeros([2,num_exp,num_epoch + 1])
for i in tqdm(range(num_exp)):
    alph = alph_all[i]
    
    temp = np.zeros([10,num_epoch+1])
    np.random.seed(i)
    for k in range(10):
        _,temp[k,:] = run_experiment(T,alph)
    cur_los[0,i,:] = np.mean(temp,0)
    cur_los[1,i,:] = np.std(temp,0)/np.sqrt(10)


#%%

epochs = [1000, 2000, 3000, 5000, 10000]

cmap = plt.cm.viridis   # you can swap this for plasma, inferno, magma, cividis
colors = cmap(np.linspace(0.1, 0.9, len(epochs)))

for ep, col in zip(epochs, colors):
    plt.errorbar(
        alph_all,
        cur_los[0, :, ep],
        cur_los[1, :, ep],
        label=str(ep),
        color=col
    )

plt.xscale('log')
plt.yscale('log')
plt.legend(title='Epoch')
plt.xlabel('Learning rate')
plt.ylabel('Loss value')

plt.savefig('Fig2d.pdf')
plt.show()




