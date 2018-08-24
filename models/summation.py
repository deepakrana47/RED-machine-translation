import numpy as np

def sum_vector(x):
    h_vect = x['h_vect']
    vect = x['vects']
    wsize = x['w_size']
    svect={}
    for hv in range(len(h_vect)):
        s = 0
        for j in h_vect[hv]:
            s += vect[j]
        vect[wsize+hv] = s
        svect[hv] = s
    return vect