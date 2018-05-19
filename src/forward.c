#include <stdio.h>
#include <math.h>
#include <shalw.h>
#include <export.h>
#include <immintrin.h> 
#include <malloc.h>

inline __m256d hFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //HPHY(t - 1, i, j) est encore nul
  __m256d hp, hp1, hf1, zh1, zh2;
  hp = _mm256_load_pd(&HPHY(t,i,j));
  hp1=_mm256_load_pd(&HPHY(t-1,i,j));
  hf1 = _mm256_load_pd(&HFIL(t-1,i,j));
  zh1 = _mm256_set_pd(alpha, alpha, alpha, alpha);
  zh2 = _mm256_set_pd(2, 2 ,2 ,2);
  if (t <= 2)
    return hp;
//    return HPHY(t - 1, i, j) + alpha * (HFIL(t - 1, i, j) - 2 * HPHY(t - 1, i, j) + HPHY(t, i, j));
  return _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(zh1,hf1),hp1),_mm256_add_pd(_mm256_mul_pd(zh2,hp1),hp));
}

inline __m256d uFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //UPHY(t - 1, i, j) est encore nul
  __m256d up, up1, uf1, yh1, yh2;
  up = _mm256_load_pd(&UPHY(t,i,j));
  up1=_mm256_load_pd(&UPHY(t-1,i,j));
  uf1 = _mm256_load_pd(&UFIL(t-1,i,j));
  yh1 = _mm256_set_pd(alpha, alpha, alpha, alpha);
  yh2 = _mm256_set_pd(2, 2 ,2 ,2);
  if (t <= 2)
    return up;

  //return UPHY(t - 1, i, j) +  alpha * (UFIL(t - 1, i, j) - 2 * UPHY(t - 1, i, j) + UPHY(t, i, j));
  return _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(yh1,uf1),_mm256_add_pd(_mm256_mul_pd(yh2,up1),up)),up1);
}

inline __m256d vFil_forward(int t, int i, int j) {
  //Phase d'initialisation du filtre
  //VPHY(t - 1, i, j) est encore nul
  __m256d vp, vp1, vf1, xh1, xh2;
  vp = _mm256_load_pd(&VPHY(t,i,j));
  vp1=_mm256_load_pd(&VPHY(t-1,i,j));
  vf1 = _mm256_load_pd(&VFIL(t-1,i,j));
  xh1 = _mm256_set_pd(alpha, alpha, alpha, alpha);
  xh2 = _mm256_set_pd(2, 2 ,2 ,2);
  if (t <= 2)
    return vp;
  //return VPHY(t - 1, i, j) + alpha * (VFIL(t - 1, i, j) - 2 * VPHY(t - 1, i, j) + VPHY(t, i, j));
  return _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(xh1,vf1), _mm256_add_pd(_mm256_mul_pd(xh2,vp1),vp)),vp1);
}

inline __m256d hPhy_forward(int t, int i, int j) {
 // double c, d;
  __m256d up120, vp201, c, d, dt1, dx1, dy1, hmoy1, up1, vp1, hf1;
  up1=_mm256_load_pd(&UPHY(t-1,i,j));
  vp1=_mm256_load_pd(&VPHY(t-1,i,j));
  hf1=_mm256_load_pd(&HFIL(t-1,i,j));
  up120=_mm256_load_pd(&UPHY(t-1,i-1,j));
  vp201=_mm256_load_pd(&VPHY(t-1,i,j+4));
  c = _mm256_set_pd(0, 0, 0, 0);
  d = _mm256_set_pd(0, 0, 0, 0);
  dt1 = _mm256_set_pd(dt, dt, dt, dt);
  dx1 = _mm256_set_pd(dx, dx, dx, dx);
  dy1 = _mm256_set_pd(dy, dy, dy, dy);
  hmoy1 = _mm256_set_pd(hmoy, hmoy, hmoy, hmoy);

  //c = 0.;
  if (i > 0)
    c = up120;

  //d = 0.;
  if (j < size_y - 1)
    d = vp201;
  //return HFIL(t - 1, i, j) - dt * hmoy * ((UPHY(t - 1, i, j) - c) / dx +  (d - VPHY(t - 1, i, j)) / dy);
  return _mm256_sub_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(dt1,hmoy1),_mm256_div_pd(_mm256_sub_pd(up1,c), dx1)), _mm256_div_pd(_mm256_sub_pd(d,vp1),dy1)),hf1);
}

inline __m256d uPhy_forward(int t, int i, int j) {
  //double b, e, f, g;
  __m256d hp120, vp102, vp120, vp122, a, b, e, f, g, q, dt1, dx1, dissip1, grav1, pcor1, hp1, vp1, uf1;
  hp1=_mm256_load_pd(&HPHY(t-1,i,j));
  vp1=_mm256_load_pd(&VPHY(t-1,i,j));
  uf1=_mm256_load_pd(&UFIL(t-1,i,j));

  hp120=_mm256_load_pd(&HPHY(t-1,i+1,j));
  vp102=_mm256_load_pd(&VPHY(t-1,i,j+4));
  vp120=_mm256_load_pd(&VPHY(t-1,i+1,j));
  vp122=_mm256_load_pd(&VPHY(t-1,i+1,j+4));
  b = _mm256_set_pd(0, 0, 0, 0);
  e = _mm256_set_pd(0, 0, 0, 0);
  f = _mm256_set_pd(0, 0, 0, 0);
  g = _mm256_set_pd(0, 0, 0, 0);
  q = _mm256_set_pd(4, 4, 4, 4);
  dt1 = _mm256_set_pd(dt, dt, dt, dt);
  dx1 = _mm256_set_pd(dx, dx, dx, dx);
  dissip1 = _mm256_set_pd( dissip, dissip, dissip, dissip);
  grav1 = _mm256_set_pd(-grav, -grav, -grav, -grav);
  pcor1 = _mm256_set_pd(pcor, pcor, pcor, pcor);
  a = _mm256_set_pd(0, 0, 0, 0);

  if (i == size_x - 1)
    return a;

//  b = 0.;
  if (i < size_x - 1)
    b = hp120;

//  e = 0.;
  if (j/4 < size_y - 1)
    e = vp102;

//  f = 0.;
  if (i < size_x - 1)
    f = vp120;

//  g = 0.;
  if (i < size_x - 1 && j < size_y - 1)
    g = vp122;

  //return UFIL(t - 1, i, j) + dt * ((-grav / dx) * (b - HPHY(t - 1, i, j)) + (pcor / 4.) * (VPHY(t - 1, i, j) + e + f + g) -(dissip * UFIL(t - 1, i, j)));
  return _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_div_pd(grav1,dx1),dt1),_mm256_sub_pd(b,hp1)),_mm256_sub_pd(_mm256_mul_pd(_mm256_div_pd(pcor1, q), _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(vp1,e),f),g)), _mm256_mul_pd(dissip1, uf1))), uf1);

}

inline __m256d vPhy_forward(int t, int i, int j) {
  //double c, d, e, f;
  
  __m256d hp101, up111, up110, up102, a, c, d, e, f, q, dt1, dy1, dissip1, grav1, pcor1, hp1, up1, vf1;
  hp1=_mm256_load_pd(&HPHY(t-1,i,j));
  up1=_mm256_load_pd(&UPHY(t-1,i,j));
  vf1=_mm256_load_pd(&VFIL(t-1,i,j));

  hp101=_mm256_load_pd(&HPHY(t-1,i,j-4));
  up111=_mm256_load_pd(&UPHY(t-1,i-1,j-4));
  up110=_mm256_load_pd(&UPHY(t-1,i-1,j));
  up102=_mm256_load_pd(&UPHY(t-1,i,j-4));
  c = _mm256_set_pd(0, 0, 0, 0);
  d = _mm256_set_pd(0, 0, 0, 0);
  e = _mm256_set_pd(0, 0, 0, 0);
  f = _mm256_set_pd(0, 0, 0, 0);
  q = _mm256_set_pd(4, 4, 4, 4);
  dt1 = _mm256_set_pd(dt, dt, dt, dt);
  dy1 = _mm256_set_pd(dy, dy, dy, dy);
  dissip1 = _mm256_set_pd( dissip, dissip, dissip, dissip);
  grav1 = _mm256_set_pd(-grav, -grav, -grav, -grav);
  pcor1 = _mm256_set_pd(pcor, pcor, pcor, pcor);
  a = _mm256_set_pd(0, 0, 0, 0);

  if (j == 0)
    return a;

  //c = 0.;
  if (j/4 > 0)
    c = hp101;

 // d = 0.;
  if (i > 0 && j/4 > 0)
    d = up111;

 // e = 0.;
  if (i > 0)
    e = up110;

 // f = 0.;
  if (j/4 > 0)
    f = up102;
 //return VFIL(t - 1, i, j) + dt * ((-grav / dy) * (HPHY(t - 1, i, j) - c) - (pcor / 4.) * (d + e + f + UPHY(t - 1, i, j)) -  (dissip * VFIL(t - 1, i, j)));
  
  return _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_div_pd(grav1,dy1),dt1),_mm256_sub_pd(hp1,c)),_mm256_sub_pd(_mm256_mul_pd(_mm256_div_pd(pcor1, q), _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(up1,d),e),f)), _mm256_mul_pd(dissip1, vf1))), vf1);
}

void forward(void) {
  FILE *file = NULL;
  double svdt = 0.;
  int t = 0;
  
  if (file_export) {
    file = create_file();
    export_step(file, t);
  }
    
  for (t = 1; t < nb_steps; t++) {
    if (t == 1) {
      svdt = dt;
      dt = 0;
    }
    if (t == 2){
      dt = svdt / 2.;
    }
     
    __m256d hp, up, vp, hf, uf, vf;
    
    for (int i = 0; i < size_x; i++) {
      for (int j = 0; j < size_y; j+=4) {
        //hp = _mm256_load_pd(&HPHY(t,i,j));
        //up = _mm256_load_pd(&UPHY(t,i,j));
       // vp = _mm256_load_pd(&VPHY(t,i,j));
       // hf = _mm256_load_pd(&HFIL(t,i,j));
       // uf = _mm256_load_pd(&UFIL(t,i,j));
       // vf = _mm256_load_pd(&VFIL(t,i,j));

      	hp = hPhy_forward(t, i, j);
      	up = uPhy_forward(t, i, j);
      	vp = vPhy_forward(t, i, j);
      	hf = hFil_forward(t, i, j);
      	uf = uFil_forward(t, i, j);
      	vf = vFil_forward(t, i, j);

        _mm256_store_pd(&HPHY(t,i,j),hp);
        _mm256_store_pd(&UPHY(t,i,j),up);
        _mm256_store_pd(&VPHY(t,i,j),vp);
        _mm256_store_pd(&HFIL(t,i,j),hf);
        _mm256_store_pd(&UFIL(t,i,j),uf);
        _mm256_store_pd(&VFIL(t,i,j*4),vf);
      }
    }

    if (file_export) {
      export_step(file, t);
    }
    
    if (t == 2) {
      dt = svdt;
    }
  }

  if (file_export) {
    finalize_export(file);
  }
}
