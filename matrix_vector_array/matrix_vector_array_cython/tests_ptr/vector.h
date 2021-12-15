typedef struct _Vector Vector;

Vector *vector_new( double *data, int length);
Vector vector_alloc();
void vector_free(Vector *vec);
Vector *add_vector(Vector *x, Vector* y);
Vector *add_scalar(Vector *x, double scal);
