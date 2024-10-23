'''

Based spike code off of 
https://github.com/ooici/ion-functions/tree/master/ion_functions/qc

'''
import numpy as np 

def spikevalues(dat, L, N, acc):
    cdef int dat_shape = dat.shape[0]
    cdef np.ndarray[double] x = dat
    cdef np.ndarray[signed char] out = np.zeros([dat_shape], dtype=np.int8)
    out.fill(1)
    spike(&out[0], &x[0], dat_shape, L, N, acc)

    return out

def dataqc_spiketest_wrapper(dat, acc, N, L, strict_validation=False):
    if is_none(acc) or is_fill(acc) or is_none(N) or is_fill(N) or is_none(L) or is_fill(L):
        out = np.empty(dat.shape, dtype=np.int8)
        out.fill(-99)
        return out
    return dataqc_spiketest(dat, np.atleast_1d(acc)[-1], np.atleast_1d(N)[-1], np.atleast_1d(L)[-1], strict_validation=strict_validation)

def dataqc_spiketest(dat, acc, N=5, L=5, strict_validation=False):
    """
    Description:

        Data quality control algorithm testing a time series for spikes.
        Returns 1 for presumably good data and 0 for data presumed bad.

        The time series is divided into windows of len L (an odd integer
        number). Then, window by window, each value is compared to its (L-1)
        neighboring values: a range R of these (L-1) values is computed (max.
        minus min.), and replaced with the measurement accuracy ACC if ACC>R. A
        value is presumed to be good, i.e. no spike, if it deviates from the
        mean of the (L-1) peers by less than a multiple of the range,
        N*max(R,ACC).

        Further than (L-1)/2 values from the start or end points, the peer
        values are symmetrically before and after the test value. Within that
        range of the start and end, the peers are the first/last L values
        (without the test value itself).

        The purpose of ACC is to restrict spike detection to deviations
        exceeding a minimum threshold value (N*ACC) even if the data have
        little variability. Use ACC=0 to disable this behavior.

    Implemented by:

        2012-07-28: DPS authored by Mathias Lankhorst. Example code provided
        for Matlab.
        2013-04-06: Christopher Wingard. Initial python implementation.
        2013-05-30: Christopher Mueller. Performance optimizations.

    Usage:

        qcflag = dataqc_spiketest(dat, acc, N, L)

            where

        qcflag = Boolean, 0 if value is outside range, else = 1.

        dat = input data set, a numeric, real vector.
        acc = Accuracy of any input measurement.
        N = (optional, defaults to 5) Range multiplier, cf. above
        L = (optional, defaults to 5) Window len, cf. above

    References:

        OOI (2012). Data Product Specification for Spike Test. Document
            Control Number 1341-10006. https://alfresco.oceanobservatories.org/
            (See: Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-10006_Data_Product_SPEC_SPKETST_OOI.pdf)
    """
    dat = np.atleast_1d(dat)

    if strict_validation:
        if not utils.isnumeric(dat).all():
            raise ValueError('\'dat\' must be numeric')

        if not utils.isreal(dat).all():
            raise ValueError('\'dat\' must be real')

        if not utils.isvector(dat):
            raise ValueError('\'dat\' must be a vector')

        for k, arg in {'acc': acc, 'N': N, 'L': L}.iteritems():
            if not utils.isnumeric(arg).all():
                raise ValueError('\'{0}\' must be numeric'.format(k))

            if not utils.isreal(arg).all():
                raise ValueError('\'{0}\' must be real'.format(k))
    dat = np.asanyarray(dat, dtype=np.float)
    
    out = spikevalues(dat, L, N, acc)
    return out
