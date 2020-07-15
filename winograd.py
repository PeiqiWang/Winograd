import numpy as np
from sympy import symbols, Matrix, Poly, zeros, eye, Indexed, simplify, IndexedBase, init_printing, pprint, Rational
from operator import mul
from functools import reduce

def At(a,m,n):
    return Matrix(m, n, lambda i,j: a[i]**j)

def A(a,m,n):
    return At(a, m-1, n).row_insert(m-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def T(a,n):
    return Matrix(Matrix.eye(n).col_insert(n, Matrix(n, 1, lambda i,j: -a[i]**n)))

def Lx(a,n):
    x=symbols('x')
    return Matrix(n, 1, lambda i,j: Poly((reduce(mul, ((x-a[k] if k!=i else 1) for k in range(0,n)), 1)).expand(basic=True), x))

def F(a,n):
    return Matrix(n, 1, lambda i,j: reduce(mul, ((a[i]-a[k] if k!=i else 1) for k in range(0,n)), 1))

def Fdiag(a,n):
    f=F(a,n)
    return Matrix(n, n, lambda i,j: (f[i,0] if i==j else 0))

def FdiagPlus1(a,n):
    f = Fdiag(a,n-1)
    f = f.col_insert(n-1, zeros(n-1,1))
    f = f.row_insert(n-1, Matrix(1,n, lambda i,j: (1 if j==n-1 else 0)))
    return f

def L(a,n):
    lx = Lx(a,n)
    f = F(a, n)
    return Matrix(n, n, lambda i,j: lx[i, 0].nth(j)/f[i]).T

def Bt(a,n):
    return L(a,n)*T(a,n)

def B(a,n):
    return Bt(a,n-1).row_insert(n-1, Matrix(1, n, lambda i,j: 1 if j==n-1 else 0))

def cookToomFilter(a,n,r):
    alpha = n+r-1
    f = FdiagPlus1(a,alpha)
    if f[0,0] < 0:
        f[0,:] *= -1
    AT = A(a,alpha,n).T
    G = (A(a,alpha,r).T/f).T
    BT = f * B(a,alpha).T

    return (AT,G,BT)

def winograd(input,filter,n,r,s):
    # Set up fraction interpolation points a
    a = [0,1,-1,2,-2]
    if n > 4:
        for i in range((n-3)//2):
            a.append(Rational(1,(i+1)*2))
            a.append(-Rational(1,(i+1)*2))

    AT,G,BT = cookToomFilter(a,n,r)

    for i in range (n-1, -1, -1):
        if i % s != 0:
            AT.row_del(i)
 
    #print("F(",n,"x",n,",",r,"x",r,")")
    print("AT = ")
    pprint(AT)
    print('\n')

    print("G = ")
    pprint(G)
    print('\n')

    print("BT = ")
    pprint(BT)
    print('\n')

    array_AT = np.array(AT).astype(np.float32)
    array_G = np.array(G).astype(np.float32)
    array_BT = np.array(BT).astype(np.float32)

    U = np.matmul(np.matmul(array_G,filter), array_G.transpose())
    V = np.matmul(np.matmul(array_BT,input), array_BT.transpose())

    output = np.matmul(np.matmul(array_AT, U*V), array_AT.transpose())

    return output

