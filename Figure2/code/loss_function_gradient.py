import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr




def err_func(r,T):
    if r<=0:
        return 1
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
    num_epoch = 3000
    r_cur = np.zeros([num_epoch+1])
    cur_los = np.zeros([num_epoch+1])

    r_cur[0] = 10*r_opt  
    cur_los[0] = err_func(r_cur[0],T);
    for i in range(num_epoch):
        cur_grad = gradient_loss(r_cur[i], T)
        r_cur[i+1] = r_cur[i] - alph * cur_grad
        cur_los[i+1] = err_func(r_cur[i+1],T);
    return r_cur, cur_los


x0 = 0;
T = 100;
alph = 1e-10
r_cur, cur_los = run_experiment(T,alph)
print(r_cur[0],r_cur[-1])
alph = 1e-9
r_cur2, cur_los2 = run_experiment(T,alph)

plt.plot(cur_los)
plt.plot(cur_los2)
plt.savefig('loss_decrease.pdf')
plt.show()

#%%
r_opt = np.pi**2/(4*T**2)

plt.plot(r_cur)
plt.plot(r_cur2)
plt.axhline(r_opt,color = 'black',ls = '-')
plt.axhline(r_opt/4,color = 'black',ls = '--')
plt.yscale('log')
plt.savefig('r_changes.pdf')




