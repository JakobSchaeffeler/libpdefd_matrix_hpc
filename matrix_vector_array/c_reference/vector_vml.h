#include <mkl.h>
typedef struct _Vector_vml{
    int length;
    double* data;
} Vector_vml;

Vector_vml vector_new_vml( double* data, int length);
Vector_vml vector_alloc_vml(void);
void set_length_vml(Vector_vml vec, int length);
void set_data_vml(Vector_vml vec, double* data);
double *get_data_vml(Vector_vml vec);

void copy_vml(Vector_vml src, Vector_vml dst);


void print_vec_vml(Vector_vml vec);
int get_length_vml(Vector_vml vec);
void vector_free_vml(Vector_vml vec);
double* malloc_vector_vml(int size, int i);
void dealloc_vector_vml(double* a);


double *add_vector_vml(Vector_vml x, Vector_vml y);
double *add_scalar_vml(Vector_vml x, double scal);
void iadd_scalar_vml(Vector_vml x, double scal);
void iadd_vector_vml(Vector_vml x, Vector_vml y);



double* mul_vector_vml(Vector_vml x, Vector_vml y);
double* mul_scalar_vml(Vector_vml x, double scal);
void imul_scalar_vml(Vector_vml x, double scal);
void imul_vector_vml(Vector_vml x, Vector_vml y);

double* div_vector_vml(Vector_vml x, Vector_vml y);
double* div_scalar_vml(Vector_vml x, double scal);

double* rdiv_vector_vml(Vector_vml x, Vector_vml y);
double* rdiv_scalar_vml(Vector_vml x, double scal);
void idiv_scalar_vml(Vector_vml x, double scal);
void idiv_vector_vml(Vector_vml x, Vector_vml y);
