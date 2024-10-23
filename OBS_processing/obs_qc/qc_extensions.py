

def spikevalues(dat, L, N, acc):
    cdef int dat_shape = dat.shape[0]
    cdef np.ndarray[double] x = dat
    cdef np.ndarray[signed char] out = np.zeros([dat_shape], dtype=np.int8)
    out.fill(1)
    spike(&out[0], &x[0], dat_shape, L, N, acc)

    return out