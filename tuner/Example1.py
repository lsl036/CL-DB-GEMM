import numpy as np
import matplotlib.pyplot as plt
import datetime

import GPy
import GPyOpt

from GPyOpt.methods import BayesianOptimization

bounds = np.array([[-1.0, 2.0]])
noise = 0.2

# 带了噪声的结果f ， 视作采样值
def f(X, noise=noise):
    return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)

# 初始采样点
X_init = np.array([[0.85], [1.13]])
Y_init = f(X_init)

# 拟合的范围网格 X
X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

# 待拟合的 0 噪声 标准值 Y 在 X 上的结果
Y = f(X,0)

name_list = ["target", "Baysian_Result"]
mkfile_time = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d%H%M%S')

name_list[0] = name_list[0]+mkfile_time+'.pdf'
name_list[1] = name_list[1]+mkfile_time+'.pdf'

# Plot optimization objective with noise level 
plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
plt.legend();

plt.savefig(name_list[0])
plt.show()

kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
bds = [{'name': 'X', 'type': 'continuous', 'domain': bounds.ravel()}]

optimizer = BayesianOptimization(f=f, 
                                 domain=bds,
                                 model_type='GP',
                                 kernel=kernel,
                                 acquisition_type ='EI',
                                 acquisition_jitter = 0.01,
                                 X=X_init,
                                 Y=-Y_init,
                                 noise_var = noise**2,
                                 exact_feval=False,
                                 normalize_Y=False,
                                 maximize=True)

optimizer.run_optimization(max_iter=10)
optimizer.plot_acquisition()

plt.savefig(name_list[1])
