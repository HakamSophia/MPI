#include <stdlib.h>
#include <shalw.h>
#include <immintrin.h> 
#include <malloc.h>

void alloc() {
  hFil = (double *) aligned_alloc(32,2*size_x*size_y*sizeof(double)) ;
  uFil = (double *) aligned_alloc(32,2*size_x*size_y*sizeof(double)) ;
  vFil = (double *) aligned_alloc(32,2*size_x*size_y*sizeof(double)) ;
  hPhy = (double *) aligned_alloc(32,2*size_x*size_y*sizeof(double)) ;
  uPhy = (double *) aligned_alloc(32,2*size_x*size_y*sizeof(double)) ;
  vPhy = (double *) aligned_alloc(32,2*size_x*size_y*sizeof(double)) ;
}

void dealloc(void) {
  free(hFil);
  free(uFil);
  free(vFil);
  free(hPhy);
  free(uPhy);
  free(vPhy);
}
