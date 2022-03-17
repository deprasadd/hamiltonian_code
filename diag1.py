import numpy as np
import scipy.optimize as sopt

from numpy import linalg as LA
import matplotlib.pyplot as plt



t=-1
t1 = -1
t2 = -2
N = 100


def hamiltonian_pack(t1,t2, N):
    H = t*np.add(np.eye(N, k=1), np.eye(N, k = -1))
   
    for i in range(N):
        if i % 2 == 0:
            
            H[i-1][i] = t1
            H[i][i-1] = t2
        else:
            H[i-1][i] = t2
            
            H[i][i-1] = t1
       
    return H
    
 
x = hamiltonian_pack(t1,t2, N)



e , v = LA.eigh(x)


def cosine(x, a, b):
    y = -1*a*np.cos(b*x)
    return y

def split_list(a_list):
        half = len(a_list)//2
        return a_list[:half], a_list[half:]



e1, e2 = split_list(e)
x, y = split_list(x)
x = np.arange(len(e))
y = np.arange(len(e2))



p,pc = sopt.curve_fit(cosine, x, e ,p0 = (0,np.pi/N))#2*np.pi/N))
#p1,pc1 = sopt.curve_fit(cosine, y, e2 ,p0 = (0,np.pi/N))#2*np.pi/N))



plt.figure()
plt.plot(e , 'x') 


#print(e )
plt.plot(x, cosine(x, *p))


#print(x)
#print(cosine(x, *p))
plt.title('Eigenvalues vs Energies')
plt.xlabel('Energy')
plt.ylabel('Wavelength')


plt.savefig('filename.png', dpi=300)
