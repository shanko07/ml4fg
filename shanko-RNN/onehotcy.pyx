import numpy as np
np.get_include() # do we need this on colab? 
cimport cython
cimport numpy as np

# Adding Z here so we can encode the separator
cdef dict bases={ 'A':<int>0, 'C':<int>1, 'G':<int>2, 'T':<int>3, 'Z':<int>4} 

@cython.boundscheck(False)
def one_hot( str string ):
    cdef np.ndarray[np.float32_t, ndim=2] res = np.zeros( (5,len(string)), dtype=np.float32 )
    cdef int j
    for j in range(len(string)):
        if string[j] in bases: # bases can be 'N' signifying missing: this corresponds to all 0 in the encoding
            res[ bases[ string[j] ], j ]=float(1.0)
    return(res)