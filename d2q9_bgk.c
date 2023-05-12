#include "d2q9_bgk.h"
#include <immintrin.h>
#include <omp.h>

/*zxx test-5.8*/
/* The main processes in one step */
int collision(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells, int* obstacles);
int streaming(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells);
int obstacle(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells, int* obstacles);
int boundary(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells, float* inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells, float* inlets, int* obstacles)
{
    /* The main time overhead, you should mainly optimize these processes. */
    /*int obstacle_num=0;
    for(int i=0;i< params.nx*params.ny;i++){
      if(obstacles[i]==1){
        obstacle_num++;
      }
    }
  printf("total:%d， obstacle:%d\n", params.nx*params.ny,obstacle_num);*/
    collision(params, cells, tmp_cells, obstacles);

    obstacle(params, cells, tmp_cells, obstacles);
    streaming(params, cells, tmp_cells);
    boundary(params, cells, tmp_cells, inlets);
  /*for(int i=0;i<NSPEEDS;i++){
    printf("tmpcells[%d].speeds[0]=%f\n",i,tmp_cells[i].cells[0]);
  }*/
    return EXIT_SUCCESS;
}

/*
** The collision of fluids in the cell is calculated using
** the local equilibrium distribution and relaxation process
*/
const float c_sq = 1.f / 3.f; /* square of speed of sound */
const float w0 = 4.f / 9.f;   /* weighting factor */
const float w1 = 1.f / 9.f;   /* weighting factor */
const float w2 = 1.f / 36.f;  /* weighting factor */

int collision(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells, int* obstacles) {
  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  const __m256 zero_vec=_mm256_setzero_ps();
  const __m256 one_vec=_mm256_set1_ps(1.f);
  const __m256 c_sq_vec=_mm256_set1_ps(c_sq);
  const __m256 w0_vec=_mm256_set1_ps(w0);
  const __m256 w1_vec=_mm256_set1_ps(w1);
  const __m256 w2_vec=_mm256_set1_ps(w2);
  #pragma omp parallel for schedule(static)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii+=SIMDLEN)
    {
      const int pos = ii + jj*params.nx;
      __m256 data[NSPEEDS]={_mm256_load_ps(&cells[0].cells[pos]),_mm256_load_ps(&cells[1].cells[pos]),_mm256_load_ps(&cells[2].cells[pos]),
                            _mm256_load_ps(&cells[3].cells[pos]),_mm256_load_ps(&cells[4].cells[pos]),
                            _mm256_load_ps(&cells[5].cells[pos]),_mm256_load_ps(&cells[6].cells[pos]),
                            _mm256_load_ps(&cells[7].cells[pos]),_mm256_load_ps(&cells[8].cells[pos])};

      __m256i obstacle_mask=_mm256_xor_si256(_mm256_load_si256((__m256i *)&obstacles[pos]),_mm256_set1_epi32(1));
      /* __m256i obstacle_mask=_mm256_load_si256((__m256i *)&obstacles[pos]); */
/*       int tmp[9];
      _mm256_storeu_si256(tmp,obstacle_mask);
        for(int i=0;i<8;i++){
            printf("obs=%d",obstacles[pos]);
            printf("tmp[%d]=%d\n",i,tmp[i]);
        } */

      if (!_mm256_testz_si256(obstacle_mask,obstacle_mask)){
        /* compute local density total */
        /* float local_density = 0.f; */
        __m256 local_density_vec = _mm256_setzero_ps();
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
            local_density_vec = _mm256_add_ps(local_density_vec, data[kk]);
        }

        /* compute x velocity component */
        /* float u_x = (cells[pos].cells[1]
                      + cells[pos].cells[5]
                      + cells[pos].cells[8]
                      - (cells[pos].cells[3]
                         + cells[pos].cells[6]
                         + cells[pos].cells[7]))
                     / local_density; */
        __m256 u_x_vec =_mm256_div_ps(
                            _mm256_sub_ps(
                                _mm256_add_ps(
                                    _mm256_add_ps(data[1],data[5]),
                                    _mm256_sub_ps(data[8],data[3])),
                                _mm256_add_ps(data[6],data[7])),
                            local_density_vec);
        /* compute y velocity component */
        /* float u_y = (cells[pos].cells[2]
                      + cells[pos].cells[5]
                      + cells[pos].cells[6]
                      - (cells[pos].cells[4]
                         + cells[pos].cells[7]
                         + cells[pos].cells[8]))
                     / local_density; */
        __m256 u_y_vec =_mm256_div_ps(
                            _mm256_sub_ps(
                                _mm256_add_ps(
                                    _mm256_add_ps(data[2],data[5]),
                                    _mm256_sub_ps(data[6],data[4])),
                                _mm256_add_ps(data[7],data[8])),
                            local_density_vec);
        /* velocity squared */
        /* float u_sq = u_x * u_x + u_y * u_y; */
        __m256 u_sq_vec= _mm256_add_ps(_mm256_mul_ps(u_x_vec,u_x_vec),_mm256_mul_ps(u_y_vec,u_y_vec));
        /* directional velocity components */
        
        /* float u[NSPEEDS]; */
        /* u[0] = 0;           */  /* zero */
        /* u[1] =   u_x;       */  /* east */
        /* u[2] =         u_y; */  /* north */
        /* u[3] = - u_x;       */  /* west */
        /* u[4] =       - u_y; */  /* south */
        /* u[5] =   u_x + u_y; */  /* north-east */
        /* u[6] = - u_x + u_y; */  /* north-west */
        /* u[7] = - u_x - u_y; */  /* south-west */
        /* u[8] =   u_x - u_y; */  /* south-east */
        /*TODO: Do above using simd*/
        __m256 u_vec[NSPEEDS];
        u_vec[0]=zero_vec;
        u_vec[1]=u_x_vec;
        u_vec[2]=u_y_vec;
        u_vec[3]=_mm256_sub_ps(zero_vec,u_x_vec);
        u_vec[4]=_mm256_sub_ps(zero_vec,u_y_vec);
        u_vec[5]=_mm256_add_ps(u_x_vec,u_y_vec);
        u_vec[6]= _mm256_sub_ps(u_y_vec,u_x_vec);
        u_vec[7]= _mm256_add_ps(u_vec[3],u_vec[4]);
        u_vec[8]= _mm256_sub_ps(u_x_vec,u_y_vec);

        /* equilibrium densities */
        /* float d_equ[NSPEEDS]; */
        /* zero velocity density: weight w0 */

        
        /* axis speeds: weight w1 */
        /* d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq)); */
        /* diagonal speeds: weight w2 */
        /* d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq)); */
        /* use simd */
        /* const float two_c_sq=2.f*c_sq;
        const float u_sq_2_c_sq=u_sq/two_c_sq;
        const float two_csq_csq=two_c_sq*c_sq; */

        const __m256 two_c_sq_vec=_mm256_set1_ps(2.f*c_sq);
        const __m256 u_sq_2_c_sq_vec=_mm256_div_ps(u_sq_vec,two_c_sq_vec);
        const __m256 two_csq_csq_vec=_mm256_mul_ps(two_c_sq_vec,c_sq_vec);
        
        /* equilibrium densities */
        __m256 d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */

        d_equ[0] = _mm256_mul_ps(
                         _mm256_mul_ps(w0_vec,local_density_vec),
                         _mm256_sub_ps(
                                 _mm256_add_ps(
                                         _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[0],c_sq_vec)),
                                         _mm256_div_ps(_mm256_mul_ps(u_vec[0],u_vec[0]), two_csq_csq_vec)),
                                 u_sq_2_c_sq_vec));

        /* axis speeds: weight w1 */
        __m256 w1_local= _mm256_mul_ps(w1_vec,local_density_vec);
        d_equ[1] = _mm256_mul_ps(
                        w1_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[1],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[1],u_vec[1]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));

        d_equ[2] = _mm256_mul_ps(
                        w1_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[2],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[2],u_vec[2]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));
        d_equ[3] = _mm256_mul_ps(
                        w1_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[3],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[3],u_vec[3]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));
        d_equ[4] = _mm256_mul_ps(
                        w1_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[4],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[4],u_vec[4]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));
        /* diagonal speeds: weight w2 */
        __m256 w2_local= _mm256_mul_ps(w2_vec,local_density_vec);
        d_equ[5] = _mm256_mul_ps(
                        w2_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[5],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[5],u_vec[5]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));
        d_equ[6] = _mm256_mul_ps(
                        w2_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[6],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[6],u_vec[6]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));
        d_equ[7] = _mm256_mul_ps(
                        w2_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[7],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[7],u_vec[7]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));
        d_equ[8] = _mm256_mul_ps(
                        w2_local,
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(one_vec, _mm256_div_ps(u_vec[8],c_sq_vec)),
                                        _mm256_div_ps(_mm256_mul_ps(u_vec[8],u_vec[8]), two_csq_csq_vec)),
                                u_sq_2_c_sq_vec));

        /* simd */
        /* printf("%f\n",cells[pos].speeds[1]); */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
            _mm256_store_ps(&tmp_cells[kk].cells[pos],
                                    _mm256_add_ps(data[kk],
                                        _mm256_mul_ps(
                                        _mm256_set1_ps(params.omega), 
                                    _mm256_sub_ps(d_equ[kk],data[kk]))));
        }
        /* relaxation step */
        /*for (int kk = 0; kk < NSPEEDS; kk++)
        {
          tmp_cells[pos].speeds[kk] = cells[pos].speeds[kk]+ params.omega * (d_equ[kk] - cells[pos].speeds[kk]);
        }*/
      }
    }
  }
  return EXIT_SUCCESS;
}


int obstacle(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells, int* obstacles) {

    /* loop over the cells in the grid */
#pragma omp parallel for schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int pos= ii + jj*params.nx;
            /* if the cell contains an obstacle */
            if (obstacles[pos])
            {
                /* called after collision, so taking values from scratch space
                ** mirroring, and writing into main grid */
                tmp_cells[0].cells[pos] = cells[0].cells[pos];
                tmp_cells[1].cells[pos] = cells[3].cells[pos];
                tmp_cells[2].cells[pos] = cells[4].cells[pos];
                tmp_cells[3].cells[pos] = cells[1].cells[pos];
                tmp_cells[4].cells[pos] = cells[2].cells[pos];
                tmp_cells[5].cells[pos] = cells[7].cells[pos];
                tmp_cells[6].cells[pos] = cells[8].cells[pos];
                tmp_cells[7].cells[pos] = cells[5].cells[pos];
                tmp_cells[8].cells[pos] = cells[6].cells[pos];
            }
        }
    }
    return EXIT_SUCCESS;
}


/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed_t* cells, t_speed_t* tmp_cells) {
    /* loop over _all_ cells */
#pragma omp parallel for schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int pos= ii + jj*params.nx;
            int jx = jj * params.nx;
            
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */
            int y_n = (jj + 1) % params.ny;
            int x_e = (ii + 1) % params.nx;
            int y_s = (jj == 0) ? (params.ny - 1) : (jj - 1);
            int x_w = (ii == 0) ? (params.nx - 1) : (ii - 1);
            int ynx = y_n * params.nx;
            int ysx = y_s * params.nx;
            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            cells[0]. cells[pos ] = tmp_cells[0].cells[pos]; /* central cell, no movement */
            cells[1]. cells[x_e + jx ] = tmp_cells[1].cells[pos]; /* east */
            cells[2].cells [ii  + ynx] = tmp_cells[2].cells[pos]; /* north */
            cells[3]. cells[x_w + jx ] = tmp_cells[3].cells[pos]; /* west */
            cells[4].cells [ii  + ysx] = tmp_cells[4].cells[pos]; /* south */
            cells[5].cells [x_e + ynx] = tmp_cells[5].cells[pos]; /* north-east */
            cells[6].cells [x_w + ynx] = tmp_cells[6].cells[pos]; /* north-west */
            cells[7].cells [x_w + ysx] = tmp_cells[7].cells[pos]; /* south-west */
            cells[8].cells [x_e + ysx] = tmp_cells[8].cells[pos]; /* south-east */
        }
    }

    return EXIT_SUCCESS;
}


/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane,
** the left border is the inlet of fixed speed, and
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed_t* cells,  t_speed_t* tmp_cells, float* inlets) {
    /* Set the constant coefficient */
    const __m256 cst1 =_mm256_set1_ps(2.0/3.0);
    const __m256 cst2 =_mm256_set1_ps(1.0/6.0);
    const __m256 cst3 =_mm256_set1_ps(1.0/2.0);

    int ii, jj;


    // top wall (bounce)
    jj = params.ny -1;
#pragma omp parallel for schedule(static)
    for(ii = 0; ii < params.nx; ii+=SIMDLEN){
        int pos= ii + jj*params.nx;
        _mm256_store_ps(&cells[4].cells[pos], _mm256_load_ps(&tmp_cells[2].cells[pos]));
        _mm256_store_ps(&cells[7].cells[pos], _mm256_load_ps(&tmp_cells[5].cells[pos]));
        _mm256_store_ps(&cells[8].cells[pos], _mm256_load_ps(&tmp_cells[6].cells[pos]));
    }

    // bottom wall (bounce)
    /*jj = 0;*/
#pragma omp parallel for schedule(static)
    for(ii = 0; ii < params.nx; ii+=SIMDLEN){
        _mm256_store_ps(&cells[2].cells[ii], _mm256_load_ps(&tmp_cells[4].cells[ii]));
        _mm256_store_ps(&cells[5].cells[ii], _mm256_load_ps(&tmp_cells[7].cells[ii]));
        _mm256_store_ps(&cells[6].cells[ii], _mm256_load_ps(&tmp_cells[8].cells[ii]));
    }

    // left wall (inlet)
    /*ii = 0;*/
#pragma omp parallel for schedule(static)
    for(jj = 0; jj < params.ny; jj+=SIMDLEN){
        int pos= jj*params.nx;
        __m256 data_0=_mm256_load_ps(&cells[0].cells[pos]);
        __m256 data_2=_mm256_load_ps(&cells[2].cells[pos]);
        __m256 data_3=_mm256_load_ps(&cells[3].cells[pos]);
        __m256 data_4=_mm256_load_ps(&cells[4].cells[pos]);
        __m256 data_6=_mm256_load_ps(&cells[6].cells[pos]);
        __m256 data_7=_mm256_load_ps(&cells[7].cells[pos]);
        /*local_density = (   cells[0].cells[pos]
                          + cells[2].cells[pos]
                          + cells[4].cells[pos]
                          + 2.0 * (cells[3].cells[pos]+cells[6].cells[pos]+cells[7].cells[pos])
                        )/(1.0 - inlets[jj]);*/
      __m256 inlets_vec= _mm256_load_ps(&inlets[jj]);
      __m256 local_density= _mm256_div_ps(
                      _mm256_add_ps(
                        _mm256_add_ps(data_0,data_2),
                        _mm256_add_ps(data_4,_mm256_mul_ps(_mm256_set1_ps(2.),
                                                                    _mm256_add_ps(_mm256_add_ps(data_3,data_6),data_7)))),
                      _mm256_sub_ps(_mm256_set1_ps(1.),inlets_vec));

        /*float local_inlet = local_density*inlets[jj];
        float index_cell = cells[2].cells[pos]-cells[4].cells[pos];*/
        __m256 local_inlet= _mm256_mul_ps(local_density,inlets_vec);
        __m256 index_cell= _mm256_sub_ps(data_2,data_4);

        /*cells[1].cells[pos] = cells[3].cells[pos]
                                             + cst1*local_inlet;*/

        /*cells[5].cells[pos] = cells[7].cells[pos]
                                             - cst3*(index_cell)
                                             + cst2*local_inlet;

        cells[8].cells[pos] = cells[6].cells[pos]
                                             + cst3*(index_cell)
                                             + cst2*local_inlet;*/
      _mm256_store_ps(&cells[1].cells[pos], _mm256_add_ps(data_3,_mm256_mul_ps(cst1,local_inlet)));
      _mm256_store_ps(&cells[5].cells[pos], _mm256_add_ps(data_7,_mm256_sub_ps(_mm256_mul_ps(cst2,local_inlet),_mm256_mul_ps(cst3,index_cell))));
      _mm256_store_ps(&cells[8].cells[pos], _mm256_add_ps(data_6,_mm256_add_ps(_mm256_mul_ps(cst2,local_inlet),_mm256_mul_ps(cst3,index_cell))));

    }

    // right wall (outlet)
    ii = params.nx-1;
#pragma omp parallel for schedule(static)
  for (jj = 0; jj < params.ny; jj++) {
    /*simd*/
    for (int kk = 0; kk < NSPEEDS; kk++) {
      const int row = jj * params.nx;
      cells[kk].cells[ii + row] = cells[kk].cells[ii - 1 + row];
    }
  }
    return EXIT_SUCCESS;
}



