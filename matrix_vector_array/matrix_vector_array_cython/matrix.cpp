#include "matrix.hpp"

sparse_matrix_t add(sparse_matrix_t m_a, sparse_matrix_t m_b){
        sparse_matrix_t m_c;
        int suc = mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, m_a, 1, m_b, &m_c);
        if(suc != 0)
            perror("Matrix addition failed");
        mkl_sparse_destroy(m_a);
        return m_c;
}

void dot_add(sparse_matrix_t m, double* x, double* y){
        matrix_descr descr = matrix_descr();
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        descr.mode = SPARSE_FILL_MODE_LOWER;
        descr.diag = SPARSE_DIAG_NON_UNIT;
        suc = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, m, descr, x,
                              1.0, y);
        if(suc != 0)
            perror("dot_add failed");
        return y;

}
