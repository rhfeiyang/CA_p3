#include "d2q9_bgk.h"
#include <immintrin.h>
#include <omp.h>

/*zxx test-5.8*/
/* The main processes in one step */
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int streaming(const t_param params, t_speed* cells, t_speed* tmp_cells);
int obstacle(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles);
int boundary(const t_param params, t_speed* cells, t_speed* tmp_cells, float* inlets);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, float* inlets, int* obstacles)
{
    /* The main time overhead, you should mainly optimize these processes. */
    collision(params, cells, tmp_cells, obstacles);
    obstacle(params, cells, tmp_cells, obstacles);
    streaming(params, cells, tmp_cells);
    boundary(params, cells, tmp_cells, inlets);
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
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {
  /* loop over the cells in the grid
  ** the collision step is called before
  ** the streaming step and so values of interest
  ** are in the scratch-space grid */
  #pragma omp parallel for schedule(static) collapse(2)
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int pos = ii + jj*params.nx;
      if (!obstacles[pos]){
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[pos].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (cells[pos].speeds[1]
                      + cells[pos].speeds[5]
                      + cells[pos].speeds[8]
                      - (cells[pos].speeds[3]
                         + cells[pos].speeds[6]
                         + cells[pos].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (cells[pos].speeds[2]
                      + cells[pos].speeds[5]
                      + cells[pos].speeds[6]
                      - (cells[pos].speeds[4]
                         + cells[pos].speeds[7]
                         + cells[pos].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        
        float u[NSPEEDS];
        u[0] = 0;            /* zero */
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */
        /*TODO: Do above using simd*/

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
        const float two_c_sq=2.f*c_sq;
        const float u_sq_2_c_sq=u_sq/two_c_sq;
        const float two_csq_csq=two_c_sq*c_sq;
        
        float d_equ_0 = w0 * local_density * (1.f + u[0] / c_sq
                                         + (u[0] * u[0]) / two_csq_csq
                                         - u_sq_2_c_sq);
        __m256 two_csq_csq_vec=_mm256_set1_ps(two_csq_csq);
        __m256 u_sq_2_c_sq_vec=_mm256_set1_ps(u_sq_2_c_sq);

        __m128 w1_vec = _mm_set_ps1(w1);
        __m128 w2_vec = _mm_set_ps1(w2);
        __m256 w=_mm256_insertf128_ps(_mm256_castps128_ps256(w1_vec),w2_vec,1);
        __m256 local_density_vec = _mm256_set1_ps(local_density);
        __m256 one= _mm256_set1_ps(1.f);
        __m256 u_1_8= _mm256_load_ps(&u[1]);
        __m256 c_sq_vec= _mm256_set1_ps(c_sq);

        __m256 d_egu_vec = _mm256_mul_ps(
                _mm256_mul_ps(w,local_density_vec),
                _mm256_sub_ps(
                        _mm256_add_ps(
                                _mm256_add_ps(one, _mm256_div_ps(u_1_8,c_sq_vec)),
                                _mm256_div_ps(_mm256_mul_ps(u_1_8,u_1_8), two_csq_csq_vec)),
                        u_sq_2_c_sq_vec));

        /* simd */
        /* printf("%f\n",cells[pos].speeds[1]); */
        tmp_cells[pos].speeds[0] = cells[pos].speeds[0]+ params.omega * (d_equ_0 - cells[pos].speeds[0]);
        __m256 omega_vec=_mm256_set1_ps(params.omega);

        __m256 cells_1_8_vec=_mm256_loadu_ps(cells[pos].speeds+1);
        _mm256_storeu_ps(tmp_cells[pos].speeds+1,_mm256_add_ps(cells_1_8_vec,_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_egu_vec,cells_1_8_vec))));
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


int obstacle(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles) {

    /* loop over the cells in the grid */
#pragma omp parallel for schedule(static) collapse(2)
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
                tmp_cells[pos].speeds[0] = cells[pos].speeds[0];
                tmp_cells[pos].speeds[1] = cells[pos].speeds[3];
                tmp_cells[pos].speeds[2] = cells[pos].speeds[4];
                tmp_cells[pos].speeds[3] = cells[pos].speeds[1];
                tmp_cells[pos].speeds[4] = cells[pos].speeds[2];
                tmp_cells[pos].speeds[5] = cells[pos].speeds[7];
                tmp_cells[pos].speeds[6] = cells[pos].speeds[8];
                tmp_cells[pos].speeds[7] = cells[pos].speeds[5];
                tmp_cells[pos].speeds[8] = cells[pos].speeds[6];
            }
        }
    }
    return EXIT_SUCCESS;
}


/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
int streaming(const t_param params, t_speed* cells, t_speed* tmp_cells) {
    /* loop over _all_ cells */
#pragma omp parallel for schedule(static) collapse(2)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int pos= ii + jj*params.nx;
            int jx = pos-ii;
            
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
            cells[ii  + jx].speeds[0] = tmp_cells[pos].speeds[0]; /* central cell, no movement */
            cells[x_e + jx].speeds[1] = tmp_cells[pos].speeds[1]; /* east */
            cells[ii  + ynx].speeds[2] = tmp_cells[pos].speeds[2]; /* north */
            cells[x_w + jx].speeds[3] = tmp_cells[pos].speeds[3]; /* west */
            cells[ii  + ysx].speeds[4] = tmp_cells[pos].speeds[4]; /* south */
            cells[x_e + ynx].speeds[5] = tmp_cells[pos].speeds[5]; /* north-east */
            cells[x_w + ynx].speeds[6] = tmp_cells[pos].speeds[6]; /* north-west */
            cells[x_w + ysx].speeds[7] = tmp_cells[pos].speeds[7]; /* south-west */
            cells[x_e + ysx].speeds[8] = tmp_cells[pos].speeds[8]; /* south-east */
        }
    }

    return EXIT_SUCCESS;
}


/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane,
** the left border is the inlet of fixed speed, and
** the right border is the open outlet of the first-order approximation.
*/
int boundary(const t_param params, t_speed* cells,  t_speed* tmp_cells, float* inlets) {
    /* Set the constant coefficient */
    const float cst1 = 2.0/3.0;
    const float cst2 = 1.0/6.0;
    const float cst3 = 1.0/2.0;

    int ii, jj;
    float local_density;

    // top wall (bounce) + loop fusion
    jj = params.ny -1;
    #pragma omp parallel for schedule(static)
    for(ii = 0; ii < params.nx; ii++){
        int pos= ii + jj*params.nx;
        cells[pos].speeds[4] = tmp_cells[pos].speeds[2];
        cells[pos].speeds[7] = tmp_cells[pos].speeds[5];
        cells[pos].speeds[8] = tmp_cells[pos].speeds[6];
        cells[ii].speeds[2] = tmp_cells[ii].speeds[4];
        cells[ii].speeds[5] = tmp_cells[ii].speeds[7];
        cells[ii].speeds[6] = tmp_cells[ii].speeds[8];
    }

    // bottom wall (bounce)
    /*jj = 0;*/
    /*#pragma omp parallel for schedule(static)
    for(ii = 0; ii < params.nx; ii++){
        cells[ii].speeds[2] = tmp_cells[ii].speeds[4];
        cells[ii].speeds[5] = tmp_cells[ii].speeds[7];
        cells[ii].speeds[6] = tmp_cells[ii].speeds[8];
    }*/

    // left wall (inlet)
    /*ii = 0;*/
    #pragma omp parallel for schedule(static)
    for(jj = 0; jj < params.ny; jj++){
        int pos= jj*params.nx;
        local_density = ( cells[pos].speeds[0]
                          + cells[pos].speeds[2]
                          + cells[pos].speeds[4]
                          + 2.0 * (cells[pos].speeds[3]+cells[pos].speeds[6]+cells[pos].speeds[7])
                        )/(1.0 - inlets[jj]);
        float local_inlet = local_density*inlets[jj];
        float index_cell = cells[pos].speeds[2]-cells[pos].speeds[4];

        cells[pos].speeds[1] = cells[pos].speeds[3]
                                             + cst1*local_inlet;

        cells[pos].speeds[5] = cells[pos].speeds[7]
                                             - cst3*(index_cell)
                                             + cst2*local_inlet;

        cells[pos].speeds[8] = cells[pos].speeds[6]
                                             + cst3*(index_cell)
                                             + cst2*local_inlet;

    }

    // right wall (outlet)
    ii = params.nx-1;
    #pragma omp parallel for schedule(static) 
    for(jj = 0; jj < params.ny; jj++){
        int row = jj*params.nx;
        cells[ii + row].speeds[0] = cells[ii-1 + row].speeds[0];
        __m256 cells_1_8_vec=_mm256_loadu_ps(cells[ii-1 + row].speeds+1);
        _mm256_storeu_ps(cells[ii + row].speeds+1, cells_1_8_vec);
    }
    return EXIT_SUCCESS;
}



