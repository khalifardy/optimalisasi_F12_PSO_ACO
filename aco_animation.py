import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import bisect
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


#ACO 
#ACO 
class ACO:
    def __init__(self, alpha,beta,rho,n_ants,n_iterations,n_kota, ruang_solusi, funtion_obj):
        """
        Inisialisasi objek ACO (Ant Colony Optimization).

        Parameters:
        - alpha (float): Nilai alpha untuk mengontrol pengaruh feromon.
        - beta (float): Nilai beta untuk mengontrol pengaruh jarak.
        - rho (float): Nilai rho untuk mengontrol penguapan feromon.
        - n_ants (int): Jumlah semut yang akan digunakan.
        - n_iterations (int): Jumlah iterasi yang akan dilakukan.
        - n_kota (int): Jumlah kota yang ada.
        - ruang_solusi (list): Daftar kota yang akan dijadikan ruang solusi.
        - funtion_obj (function): Fungsi objektif yang akan dioptimasi.

        Attributes:
        - alpha (float): Nilai alpha untuk mengontrol pengaruh feromon.
        - beta (float): Nilai beta untuk mengontrol pengaruh jarak.
        - rho (float): Nilai rho untuk mengontrol penguapan feromon.
        - feromon (numpy.ndarray): Matriks feromon.
        - ruang_solusi (list): Daftar kota yang akan dijadikan ruang solusi.
        - function_obj (function): Fungsi objektif yang akan dioptimasi.
        - history_path (list): Daftar path yang dilalui oleh setiap semut.
        - n_iterations (int): Jumlah iterasi yang akan dilakukan.
        - global_optimum (float): Nilai optimum global.
        - global_point (list): Titik optimum global.
        - matrix_distance (numpy.ndarray): Matriks jarak antar kota.
        - n_ants (int): Jumlah semut yang akan digunakan.
        - n_kota (int): Jumlah kota yang ada.
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.feromon = np.ones((len(ruang_solusi),len(ruang_solusi)))
        self.ruang_solusi = ruang_solusi
        self.function_obj = funtion_obj
        self.history_path = [[] for _ in range(n_ants)]
        self.n_iterations = n_iterations
        self.global_optimum = math.inf
        self.global_point = None
        self.matrix_distance = self.distance()
        self.n_ants = n_ants
        self.n_kota = n_kota
    
    def distance(self):
        """
        Menghitung matriks jarak antar kota.

        Returns:
        - result (numpy.ndarray): Matriks jarak antar kota.
        """
        distance = []
        for i in range(len(self.ruang_solusi)):
            row_matrix = []
            for j in range(len(self.ruang_solusi)):
                if i !=j:
                    row_matrix.append(self.function_obj(np.array(self.ruang_solusi[j]))/self.function_obj(np.array(self.ruang_solusi[i])))
                else:
                    row_matrix.append(0)
            distance.append(row_matrix)
        
        distance_float = np.array(distance).astype(float)
        result = np.where(distance_float != 0, 1/distance_float, 0)
        
        return result
    
    def probability(self, dari,ke,feromon):
        """
        Menghitung probabilitas pemilihan kota tujuan.

        Parameters:
        - dari (int): Indeks kota asal.
        - ke (int): Indeks kota tujuan.
        - feromon (numpy.ndarray): Matriks feromon.

        Returns:
        - prob (float): Probabilitas pemilihan kota tujuan.
        """
        penyebut = sum((feromon[dari]** self.alpha)*(self.matrix_distance[dari]** self.beta))
        return ((feromon[dari,ke] ** self.alpha) * (self.matrix_distance[dari,ke] ** self.beta))/penyebut
    
    def roulette_whell(self,probs):
        """
        Memilih kota tujuan menggunakan metode roulette wheel.

        Parameters:
        - probs (dict): Dictionary probabilitas pemilihan kota tujuan.

        Returns:
        - hasil (int): Indeks kota tujuan yang terpilih.
        """
        rand = np.random.rand()
        key = [ke for ke in probs.keys()]
        cumulative_probs = [probs[k] for k in key]
        index = bisect.bisect(cumulative_probs,rand)
        if index >= len(key):
            index = len(key) - 1

        hasil = key[index]
        
        return hasil
    
    def increase_feromon(self, path):
        """
        Meningkatkan nilai feromon pada jalur yang dilalui oleh setiap semut.

        Parameters:
        - path (list): Daftar path yang dilalui oleh setiap semut.
        """
        for pth in path:
            for i in range(len(pth)-1):
                self.feromon[pth[i],pth[i+1]] += 1/self.function_obj(np.array(self.ruang_solusi[i]))
    
    def decrease_feromon(self):
        """
        Mengurangi nilai feromon pada seluruh matriks feromon.
        """
        for i in range(len(self.feromon)):
            for j in range(len(self.feromon)):
                self.feromon[i,j] = (1-self.rho)*self.feromon[i,j]
    
    def fit(self):
        """
        Melakukan proses optimisasi menggunakan algoritma ACO.
        """
        for i in range(self.n_iterations):
            posisi_0 = np.random.choice(len(self.ruang_solusi))
            hist_Path = []
            for sem in range(self.n_ants):
                semut_path = [posisi_0]
                dari = posisi_0
                kota_visit = [dari]
                feromon_individu = self.feromon.copy()
                for _ in range(self.n_kota):
                    self.history_path[sem].append(self.ruang_solusi[dari])
                    for vis in kota_visit:
                        feromon_individu[dari,vis] = 0
                    proba = 0
                    probs = {}
                    for  tuju in range(len(self.ruang_solusi)):
                        if tuju  != dari:
                            proba += self.probability(dari,tuju,feromon_individu)
                            probs[tuju] = proba
                        
                    ke = self.roulette_whell(probs)
                    while ke in kota_visit:
                        ke = self.roulette_whell(probs)
                    
                    #if self.function_obj(np.array(self.ruang_solusi[ke])) < self.function_obj(np.array(self.ruang_solusi[dari])):
                    kota_visit.append(ke)
                    semut_path.append(ke)
                    dari = ke
                hist_Path.append(semut_path)
            self.decrease_feromon()
            self.increase_feromon(hist_Path)
            
            for pos in  hist_Path:
                score = self.function_obj(np.array(self.ruang_solusi[pos[-1]]))
                if score < self.global_optimum:
                    self.global_optimum = score
                    self.global_point = self.ruang_solusi[pos[-1]]
            
            print(f'Iterasi - {i+1} : global optimum {self.global_point} score : {self.global_optimum}')
            
x_1 = [i for i in range(-10,8)]
x_2 = x_1
ruang_solusi = [[i,j] for i in x_1 for j in x_2]      
#animasi ACO
# Plot hasil running algoritma 2 secara interaktif

#animasi ACO
x = np.linspace(-10, 7, 100)
y = np.linspace(-10, 7, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([f12_function(np.array([xi, yi])) for xi, yi in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
aco3= ACO(0.1,0.1,0.1,5,9,4,ruang_solusi,f12_function)
aco3.fit()
semut_list = [[j[i]for j in aco3.history_path] for i in range(len(aco3.history_path[0]) )]
semut_array = np.array(semut_list)
points = ax.scatter(semut_array[0][:, 0], semut_array[0][:, 1], [f12_function(p) for p in semut_array[0]], color='r')



def update2(frame):
    titik_poin = semut_array[frame]
    print(titik_poin)
    points._offsets3d = (titik_poin[:, 0], titik_poin[:, 1], [f12_function(p) for p in titik_poin])
    return points,

ani = FuncAnimation(fig, update2, frames=np.arange(len(semut_array)), interval=20, repeat=False)
plt.show()