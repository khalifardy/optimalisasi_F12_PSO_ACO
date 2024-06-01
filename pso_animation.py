import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import random
sns.set()

def u(xi, a, k, m):
    if xi > a:
        return k * ((xi - a) ** m)
    elif xi < -a:
        return k * ((-xi - a) ** m)
    return 0

def f12_function(x:np.array):
    n = len(x)
    y = x+1  # Transform x by adding 1 to each component
    term1 = 10 * np.sin(np.pi * y[0])
    term2 = sum((y[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * y[1:])** 2) ))
    term3 = (y[-1] - 1) ** 2
    sum_u = sum(u(xi, 10, 100, 4) for xi in x)

    
    return np.pi / n * (term1 + term2 + term3) + sum_u


# PSO
# Write your code here

class PSO:
    def __init__(self, objective_function, n_particles, n_dimensions, w, c1, c2, n_iterations, upper_bound, lower_bound, seed=None):
        """
        Inisialisasi objek PSO.

        Parameters:
        - objective_function (function): Fungsi tujuan yang akan dioptimalkan.
        - n_particles (int): Jumlah partikel dalam populasi.
        - n_dimensions (int): Jumlah dimensi dalam ruang pencarian.
        - w (float): Faktor inersia.
        - c1 (float): Faktor kognitif.
        - c2 (float): Faktor sosial.
        - n_iterations (int): Jumlah iterasi.
        - upper_bound (float): Batas atas untuk nilai partikel.
        - lower_bound (float): Batas bawah untuk nilai partikel.
        - seed (int, optional): Seed untuk inisialisasi random. Default: None.
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.obj_function = objective_function
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.w = w
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.c1 = c1
        self.c2 = c2
        self.n_iterations = n_iterations
        self.pbest = np.zeros((n_particles, n_dimensions))
        self.gbest = np.zeros(n_dimensions)
        self.pbest_score = np.zeros(n_particles)
        self.gbest_score = math.inf
        self.particles = np.random.rand(n_particles, n_dimensions) * (self.upper_bound-self.lower_bound) + self.lower_bound
        self.velocities = np.zeros((n_particles, n_dimensions))
        self.history_particel = []
    
    def velocity(self, x):
        """
        Menghitung kecepatan partikel.

        Parameters:
        - x (ndarray): Koordinat partikel.

        Returns:
        - ndarray: Kecepatan partikel.
        """
        return self.w * x + self.c1 * np.random.rand() * (self.pbest - x) + self.c2 * np.random.rand() * (self.gbest - x)
    
    def position(self, x):
        """
        Menghitung posisi partikel.

        Parameters:
        - x (ndarray): Koordinat partikel.

        Returns:
        - ndarray: Posisi partikel.
        """
        return x + self.velocity(x)
    
    def fit(self):
        """
        Melakukan optimisasi menggunakan algoritma PSO.

        Returns:
        - ndarray: Koordinat partikel terbaik.
        """
        for i in range(self.n_iterations):
            for j in range(self.n_particles):
                score = self.obj_function(self.particles[j])
                if score < self.pbest_score[j]:
                    self.pbest_score[j] = score
                    self.pbest[j] = self.particles[j]
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest = self.particles[j]
            print(f"iterasi-{i+1} : global terbaik {self.gbest} score : {self.gbest_score} ")
            self.velocities = self.velocity(self.particles)
            self.particles = self.position(self.particles)
            self.history_particel.append(self.particles)
        return self.gbest
    


# Plot hasil running PSO  secara interaktif
#untuk bisa running lebih baik silahkan buka url : 

x = np.linspace(-10, 7, 100)
y = np.linspace(-10, 7, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([f12_function(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
pso3 = PSO(f12_function,10,2,0.1,1,0.7,20,7,-10,22)
points = ax.scatter(pso3.particles[:, 0], pso3.particles[:, 1], [f12_function(p) for p in pso3.particles], color='r')
pso3.fit()
list_poin = pso3.history_particel


def update(frame):
    titik_poin = list_poin[frame]
    print(titik_poin)
    points._offsets3d = (titik_poin[:, 0], titik_poin[:, 1], [f12_function(p) for p in titik_poin])
    return points,

ani = FuncAnimation(fig, update, frames=np.arange(len(list_poin)), interval=100,repeat=False)
plt.show()
