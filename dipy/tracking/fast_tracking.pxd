cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.direction.pmf cimport PmfGen


ctypedef int (*func_ptr)(double* point,
                         double* direction,
                         TrackingParameters params,
                         PmfGen pmf_gen) noexcept nogil


cdef int generate_tractogram_c(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               int nbr_threads,
                               StoppingCriterion sc,
                               TrackingParameters params,
                               PmfGen pmf_gen,
                               func_ptr tracker,
                               double** streamlines,
                               int* length,
                               int* status)


cdef int generate_local_streamline(double* seed,
                                   double* position,
                                   double* stream,
                                   int* stream_idx,
                                   func_ptr tracker,
                                   StoppingCriterion sc,
                                   TrackingParameters params,
                                   PmfGen pmf_gen) noexcept nogil


cdef int get_pmf(double* pmf,
                 double* point,
                 PmfGen pmf_gen,
                 double pmf_threshold,
                 int pmf_len) noexcept nogil





