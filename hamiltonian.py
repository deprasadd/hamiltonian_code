import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from heapq import nsmallest
 

 

 


t = np.linspace(-5,0, num = 6)
N = 100

 

 

 


def hamiltonian(t, N):
    H = t*np.add(np.eye(N, k=-1),np.eye(N, k = 1))
    H[0][N-1] = t
    H[N-1][0] = t
    return H

 


def cosine(x, a,b):
    y = a*np.cos(b*x)
    return y

 

def sorting_eigenvalues(e):
    oe = []
    ee = []
    for i in range(len(e)):
        if(i%2 == 0):
           oe.append(e[i])
        else:
            ee.append(e[i])
    return np.array(oe),np.array(ee)
      

 

def bandwidth(t,N):      
    
    x = hamiltonian(t, N)
    e , v = LA.eigh(x)
    #print(e)
   
     
    Oe,EE = sorting_eigenvalues(e)
    OE = Oe[::-1]
    
    E = np.concatenate((OE,EE))#np.concatenate(np.array(OE),np.array(EE))
    
    xVals = []
    yVals = []
    
    z = ' , '.join(str(v) for v in E)
    
    for i in float(z):
        xVals.append(nsmallest(i, E))
        yVals.append(i)
    print(xVals)
    print(yVals)
    #value = x[0:N:10]
    
   # value = value
    
    #print(value)
    #print(e )
    plt.plot(xVals, yVals)
    #print(x)
    #print(cosine(x, *p))
    plt.title('Energies vs momenta')
    plt.xlabel('k')
    plt.ylabel('Energy')
    #plt.xticks(value)
    #plt.axvline(x=0, linsestyle ="-", color = 'green')
    plt.savefig('graph_%0.2f.png' %t)
    plt.show()
    #return np.amax(e)-np.amin(e)

 

W = []
for i in range(len(t)):
    W.append(bandwidth(t[i], N))
    
print(W)
