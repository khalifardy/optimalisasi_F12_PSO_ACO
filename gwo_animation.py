import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import bisect

def function_f9(x:np.array):
    hasil = sum(x**2 - 10 * np.cos(2*np.pi*x) + 10)
    return hasil

class GWO:
    def __init__(self,function_obj,n_wolf, iterasi,dim,batas_atas, batas_bawah):
        self.function_obj = function_obj
        self.n_wold = n_wolf
        self.iterasi = iterasi
        self.dim = dim
        self.batas_atas = batas_atas
        self.batas_bawah = batas_bawah
        self.global_optimum_score = np.inf
        self.global_titik = None
        self.history_titik = []
    
    def init_wolf(self):
        return np.random.uniform(self.batas_bawah,self.batas_atas,(self.n_wold,self.dim))
    
    def fitness(self,x):
        return self.function_obj(x)
    
    def get_alpha_beta_delta(self, wolves):
        fitness = [self.fitness(wolf) for wolf in wolves]
        idx = np.argsort(fitness)
        return wolves[idx[0]], wolves[idx[1]], wolves[idx[2]]
    
    def get_C(self):
        return 2*np.random.rand(self.dim)
    
    def get_A(self,iterasi):
        a = 2 - 2*iterasi/self.iterasi
        return 2*a*np.random.rand(self.dim) - a
    
    def get_D(self,alpha,beta,delta,wolves):
        D_alpha = np.abs(self.get_C()*alpha - wolves)
        D_beta = np.abs(self.get_C()*beta - wolves)
        D_delta = np.abs(self.get_C()*delta - wolves)
        return D_alpha,D_beta,D_delta
    
    def update_posisi(self,alpha,beta,delta,D_alpha,D_beta,D_delta,A):
        x_alpha = alpha - A*D_alpha
        x_beta = beta - A*D_beta
        x_delta = delta - A*D_delta
        
        return (x_alpha + x_beta + x_delta)/3
    
    def fit(self):
        wolves = self.init_wolf()
        for i in range(self.iterasi):
            alpha,beta,delta = self.get_alpha_beta_delta(wolves)
            titik_wolf = []
            for idx, wolf in enumerate(wolves):  
                titik_wolf.append(wolf)
                if not (np.array_equal(wolf, alpha) or np.array_equal(wolf, beta) or np.array_equal(wolf, delta)):   
                    D_alpha,D_beta,D_delta = self.get_D(alpha,beta,delta,wolf)
                    A = self.get_A(i+1)
                    wolves[idx] = self.update_posisi(alpha,beta,delta,D_alpha,D_beta,D_delta,A)
            

                score = self.fitness(wolf)
                if score < self.global_optimum_score:
                    self.global_optimum_score = score
                    self.global_titik = wolf
            self.history_titik.append(titik_wolf)
            print(f"itereasi ke-{i+1} : titik : {self.global_titik} score : {self.global_optimum_score}")
    
    
x = np.linspace(-3, 6, 100)
y = np.linspace(-3, 6, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([function_f9(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
gwo = GWO(function_f9,5,100,2,-3,7)
gwo.fit()

wolf_list = [[j[i]for j in gwo.history_titik] for i in range(len(gwo.history_titik[0]) )]
wolf_array = np.array(wolf_list)
points = ax.scatter(wolf_array[0][:, 0], wolf_array[0][:, 1], [function_f9(p) for p in wolf_array[0]], color='r')



def update2(frame):
    titik_poin = wolf_array[frame]
    print(titik_poin)
    points._offsets3d = (titik_poin[:, 0], titik_poin[:, 1], [function_f9(p) for p in titik_poin])
    return points,

ani = FuncAnimation(fig, update2, frames=np.arange(len(wolf_array)), interval=500, repeat=False)
plt.show()