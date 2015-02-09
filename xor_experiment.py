import itertools
import math
from operator import mul    # or mul=lambda x,y:x*y
from fractions import Fraction

def bitsoncount(x):
    return bin(x).count('1')

def isomorphic_list(V):
    l = list(xrange((2**(V*V))))
    MIN = 0
    MAX = (2**(V*V))-1
    SLO = 6 #238
    NSL = 9 #273
    correct = {MIN,MAX,SLO,NSL}
    
    for f in range(2,math.factorial(V)+1):
        if f == 0:
            continue
        for i in itertools.combinations(l, f):
            skip = False
            x = bitsoncount(i[0])
            for j in i:
                if(bitsoncount(j) != x):
                    skip = True
                    break
            if(skip):
                continue
            # the equivalence class contains vertices of the same bit length. Xor them
            xor = 0
            for j in i:
                xor = xor ^ j
            if xor in correct:
                print "{0} = {1}".format(xor,i)

isomorphic_list(2)


        
