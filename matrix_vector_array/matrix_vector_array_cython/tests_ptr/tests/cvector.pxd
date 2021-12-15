cdef extern from "vector.h":
    ctypedef struct Vector:
        pass
    Vector vector_new(double* data, int length)
    void vector_free(Vector vec)
    Vector add_vector(Vector x, Vector y)
    Vector add_scalar(Vector x, double scal)
