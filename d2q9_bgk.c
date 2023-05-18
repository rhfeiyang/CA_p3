#include "d2q9_bgk.h"
#include <immintrin.h>
#include <omp.h>
#include <assert.h>
/*zxx test-5.8*/
/* The main processes in one step */
int collision_obstacle(const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, int* obstacles);
int streaming_boundary(int iteration,const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, float* inlets);
int obstacle(const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, int* obstacles);
int boundary(const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, float* inlets);
int streaming_boundary_collision(int iteration,const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, float* inlets,int* obstacles);
/*
** The main calculation methods.
** timestep calls, in order, the functions:
** collision(), obstacle(), streaming() & boundary()
*/
int timestep(const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, float* inlets, int* obstacles)
{
    /* The main time overhead, you should mainly optimize these processes. */
    /*int obstacle_num=0;
    for(int i=0;i< params.nx*params.ny;i++){
      if(obstacles[i]==1){
        obstacle_num++;
      }
    }
  printf("total:%d， obstacle:%d\n", params.nx*params.ny,obstacle_num);*/

//    obstacle(params, cells, tmp_cells, obstacles);

//    boundary(params, cells, tmp_cells, inlets);
    /*for(int i=0;i<NSPEEDS;i++){
      printf("tmp(*cells)[%d].speeds[0]=%f\n",i,(*tmp_cells)[i][0]);
    }*/
    collision_obstacle(params, cells, tmp_cells, obstacles);
    /*for(int i=0;i< times-1;++i){
      streaming_boundary_collision(params,cells, tmp_cells, inlets,obstacles);
    }*/
    streaming_boundary(params.nx-1,params, cells, tmp_cells, inlets);
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

/* cells->tmp_cells */
int collision_obstacle(const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, int* obstacles) {
    /* loop over the cells in the grid
    ** the collision step is called before
    ** the streaming step and so values of interest
    ** are in the scratch-space grid */
    const __m256 zero_vec=_mm256_setzero_ps();
    /*const __m256 one_vec=_mm256_set1_ps(1.f);*/
    __m256i one_int_vec=_mm256_set1_epi32(1);
    const __m256 half_vec=_mm256_set1_ps(0.5f);
    const __m256 three_vec=_mm256_set1_ps(3.f);
    /*const __m256 c_sq_vec=_mm256_set1_ps(c_sq);*/
    const __m256 w0_vec=_mm256_set1_ps(w0);
    const __m256 w1_vec=_mm256_set1_ps(w1);
    const __m256 w2_vec=_mm256_set1_ps(w2);
#pragma omp parallel for simd schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0,set = (jj*params.nx)/SIMDLEN; ii < params.nx; ii+=SIMDLEN, set++)
        {
            const int pos = ii + jj*params.nx;
            int ind=pos%SIMDLEN;
            /* __m256i obstacle_mask=_mm256_load_si256((__m256i *)&obstacles[pos]); */
/*       int tmp[9];
      _mm256_storeu_si256(tmp,obstacle_mask);
        for(int i=0;i<8;i++){
            printf("obs=%d",obstacles[pos]);
            printf("tmp[%d]=%d\n",i,tmp[i]);
        } */
            __m256i obstacle_mask_inv=_mm256_xor_si256(_mm256_load_si256((__m256i *)&obstacles[pos]),one_int_vec);
            if (!_mm256_testz_si256(obstacle_mask_inv,obstacle_mask_inv)){
                __m256 data[NSPEEDS]={_mm256_load_ps(&(*cells)[set].speed[0][ind]),_mm256_load_ps(&(*cells)[set].speed[1][ind]),_mm256_load_ps(&(*cells)[set].speed[2][ind]),
                                      _mm256_load_ps(&(*cells)[set].speed[3][ind]),_mm256_load_ps(&(*cells)[set].speed[4][ind]),
                                      _mm256_load_ps(&(*cells)[set].speed[5][ind]),_mm256_load_ps(&(*cells)[set].speed[6][ind]),
                                      _mm256_load_ps(&(*cells)[set].speed[7][ind]),_mm256_load_ps(&(*cells)[set].speed[8][ind])};


                /* compute local density total */
                /* float local_density = 0.f; */
                __m256 local_density_vec = _mm256_add_ps( _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(data[0],data[1]),_mm256_add_ps(data[2],data[3])),
                                                                        _mm256_add_ps(_mm256_add_ps(data[4],data[5]),_mm256_add_ps(data[6],data[7]))),data[8]);
                /* compute x velocity component */
                /* float u_x = ((*cells)[pos][1]
                              + (*cells)[pos][5]
                              + (*cells)[pos][8]
                              - ((*cells)[pos][3]
                                 + (*cells)[pos][6]
                                 + (*cells)[pos][7]))
                             / local_density; */
                __m256 u_x_vec =_mm256_div_ps(
                        _mm256_sub_ps(
                                _mm256_add_ps(
                                        _mm256_add_ps(data[1],data[5]),
                                        _mm256_sub_ps(data[8],data[3])),
                                _mm256_add_ps(data[6],data[7])),
                        local_density_vec);
                /* compute y velocity component */
                /* float u_y = ((*cells)[pos][2]
                              + (*cells)[pos][5]
                              + (*cells)[pos][6]
                              - ((*cells)[pos][4]
                                 + (*cells)[pos][7]
                                 + (*cells)[pos][8]))
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
                __m256 u_sq_vec_3=_mm256_mul_ps(u_sq_vec,_mm256_set1_ps(-3.f));
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
                u_vec[1]= _mm256_mul_ps(u_x_vec,three_vec);
                u_vec[2]=_mm256_mul_ps(u_y_vec,three_vec);
                u_vec[3]=_mm256_sub_ps(zero_vec,u_vec[1]);
                u_vec[4]=_mm256_sub_ps(zero_vec,u_vec[2]);
                u_vec[5]=_mm256_add_ps(u_vec[1],u_vec[2]);
                u_vec[6]= _mm256_sub_ps(u_vec[2],u_vec[1]);
                u_vec[7]= _mm256_add_ps(u_vec[3],u_vec[4]);
                u_vec[8]= _mm256_sub_ps(u_vec[1],u_vec[2]);

                /* equilibrium densities */
                /* float d_equ[NSPEEDS]; */
                /* zero velocity density: weight w0 */

                /*d_equ[0] = w0 * local_density * (1.f + u[0] / c_sq
                                                 + (u[0] * u[0]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));*/
                /*
                 * u[0]=0, d_equ=w0*dens*(1+0+0-3/2 * u_sq)=w0*dens*(1-3/2 * u_sq)
                 *              =w0*( dens*(-3/2 * u_sq)+dens )
                 * */
                /* axis speeds: weight w1 */
                /* d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                                 + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                                 - u_sq / (2.f * c_sq));
                 c_sq=1/3    (w1*dens) * (0.5*(3u*3u-3u_sq)+3u) + (w1*dens)
                            = (w1*dens) * (9/2 * (u*u)- 3/2 *u_sq +3u+1)


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

                /*const __m256 two_c_sq_vec=_mm256_set1_ps(2.f*c_sq);
                const __m256 u_sq_2_c_sq_vec=_mm256_div_ps(u_sq_vec,two_c_sq_vec);
                const __m256 two_csq_csq_vec=_mm256_mul_ps(two_c_sq_vec,c_sq_vec);*/

                /* equilibrium densities */
                __m256 d_equ[NSPEEDS];
                /* zero velocity density: weight w0 */
                /*u[0]=0, d_equ=w0*dens*(1+0+0-3/2 * u_sq)=w0*dens*(1-3/2 * u_sq)=w0*( dens*(-3/2 * u_sq)+dens )*/

                d_equ[0] = _mm256_mul_ps(w0_vec,_mm256_fmadd_ps(_mm256_mul_ps(u_sq_vec_3,half_vec),local_density_vec,local_density_vec));

                /*c_sq=1/3   (w1*dens) * (0.5*(3u*3u-3u_sq)+3u) + (w1*dens) = (w1*dens) * (9/2 * (u*u)- 3/2 *u_sq +3u+1)*/
                /* axis speeds: weight w1 */
                __m256 w1dens= _mm256_mul_ps(w1_vec,local_density_vec);
                d_equ[1] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[1],u_vec[1],u_sq_vec_3),half_vec,u_vec[1]),w1dens,w1dens);
                d_equ[2] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[2],u_vec[2],u_sq_vec_3),half_vec,u_vec[2]),w1dens,w1dens);
                d_equ[3] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[3],u_vec[3],u_sq_vec_3),half_vec,u_vec[3]),w1dens,w1dens);
                d_equ[4] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[4],u_vec[4],u_sq_vec_3),half_vec,u_vec[4]),w1dens,w1dens);

                /* diagonal speeds: weight w2 */
                __m256 w2dens= _mm256_mul_ps(w2_vec,local_density_vec);
                d_equ[5] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[5],u_vec[5],u_sq_vec_3),half_vec,u_vec[5]),w2dens,w2dens);
                d_equ[6] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[6],u_vec[6],u_sq_vec_3),half_vec,u_vec[6]),w2dens,w2dens);
                d_equ[7] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[7],u_vec[7],u_sq_vec_3),half_vec,u_vec[7]),w2dens,w2dens);
                d_equ[8] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[8],u_vec[8],u_sq_vec_3),half_vec,u_vec[8]),w2dens,w2dens);

                /* simd */
                /* printf("%f\n",(*cells)[pos].speeds[1]); */
                __m256 omega_vec=_mm256_set1_ps(params.omega);
                __m256 obstacle_mask=_mm256_castsi256_ps(_mm256_cmpeq_epi32(obstacle_mask_inv, one_int_vec));
                _mm256_store_ps(&(*tmp_cells)[set].speed[0][ind],_mm256_blendv_ps(data[0],_mm256_add_ps(data[0],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[0],data[0]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[1][ind],_mm256_blendv_ps(data[3],_mm256_add_ps(data[1],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[1],data[1]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[2][ind],_mm256_blendv_ps(data[4],_mm256_add_ps(data[2],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[2],data[2]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[3][ind],_mm256_blendv_ps(data[1],_mm256_add_ps(data[3],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[3],data[3]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[4][ind],_mm256_blendv_ps(data[2],_mm256_add_ps(data[4],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[4],data[4]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[5][ind],_mm256_blendv_ps(data[7],_mm256_add_ps(data[5],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[5],data[5]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[6][ind],_mm256_blendv_ps(data[8],_mm256_add_ps(data[6],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[6],data[6]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[7][ind],_mm256_blendv_ps(data[5],_mm256_add_ps(data[7],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[7],data[7]))),obstacle_mask));
                _mm256_store_ps(&(*tmp_cells)[set].speed[8][ind],_mm256_blendv_ps(data[6],_mm256_add_ps(data[8],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[8],data[8]))),obstacle_mask));

                /* relaxation step */
                /*for (int kk = 0; kk < NSPEEDS; kk++)
                {
                  (*tmp_cells)[pos].speeds[kk] = (*cells)[pos].speeds[kk]+ params.omega * (d_equ[kk] - (*cells)[pos].speeds[kk]);
                }*/
            } else{
                _mm256_store_ps(&(*tmp_cells)[set].speed[0][ind],_mm256_load_ps(&(*cells)[set].speed[0][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[1][ind],_mm256_load_ps(&(*cells)[set].speed[3][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[2][ind],_mm256_load_ps(&(*cells)[set].speed[4][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[3][ind],_mm256_load_ps(&(*cells)[set].speed[1][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[4][ind],_mm256_load_ps(&(*cells)[set].speed[2][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[5][ind],_mm256_load_ps(&(*cells)[set].speed[7][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[6][ind],_mm256_load_ps(&(*cells)[set].speed[8][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[7][ind],_mm256_load_ps(&(*cells)[set].speed[5][ind]));
                _mm256_store_ps(&(*tmp_cells)[set].speed[8][ind],_mm256_load_ps(&(*cells)[set].speed[6][ind]));
            }
        }
    }
    return EXIT_SUCCESS;
}

static inline void speed_update(t_speed_t** cells,t_speed_t ** tmp_cells,int dir,int pos_set,int neighbour_set,int x_w, int jx,int ii,int ysx,int x_e,int ynx,const __m256i* left_mask,const __m256i* right_mask){
    int tmp_pos,set,ind;
    if(dir==1) { tmp_pos = x_w + jx;}
    else if (dir==2) tmp_pos=ii  + ysx;
    else if (dir==3) { tmp_pos = x_e + jx; }
    else if (dir==4) tmp_pos=ii  + ynx;
    else if (dir==5) { tmp_pos = x_w + ysx; }
    else if (dir==6) { tmp_pos = x_e + ysx; }
    else if (dir==7) { tmp_pos = x_e + ynx; }
    else if (dir==8) { tmp_pos = x_w + ynx; }

    int source_dir=dir;
    set=tmp_pos/SIMDLEN; ind=tmp_pos%SIMDLEN;
//  printf("pos_x: %d, pos_y:%d ,dir: %d ,temp_x: %d, temp_y:%d\n",pos_set*8%4096,pos_set*8/4096,dir,tmp_pos%4096,tmp_pos/4096);
    if(dir==2 || dir==4){
        if(ynx==0 && dir==4) { source_dir = 2;set=pos_set; }
        else if(jx==0 && dir==2) { source_dir = 4; set=pos_set; }

        _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],_mm256_load_ps(&(*tmp_cells)[set].speed[source_dir][0]));
    }
    else if(dir==1 || dir==5 || dir==8){
        if(ynx==0 && dir==8) { source_dir = 6; neighbour_set=pos_set; _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]));}
        else if(jx==0 && dir==5) { source_dir = 7; neighbour_set=pos_set; _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]));}
        else{
            __m256 tmp=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);
            tmp[7]=(*tmp_cells)[set].speed[dir][ind];
            _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],
                            _mm256_permutevar8x32_ps(tmp,*left_mask));
        }
    }
    else{
        /*3 6 7*/
        if(ynx==0 && dir==7) { source_dir = 5;neighbour_set=pos_set; _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]));}
        else if(jx==0 && dir==6) { source_dir = 8; neighbour_set=pos_set; _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]));}
        else{
            __m256 tmp=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);
            tmp[0]=(*tmp_cells)[set].speed[dir][ind];
            _mm256_store_ps(&(*cells)[pos_set].speed[dir][0],
                            _mm256_permutevar8x32_ps(tmp,*right_mask));
        }
    }
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
/*tmp_cells->cells*/
int streaming_boundary(int iteration,const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, float* inlets) {
    /* loop over _all_ cells */
//  printf("%f\n",(*cells)[0].speed[5][0]);
//  omp_set_num_threads(8);
    const __m256i left_mask=_mm256_set_epi32(6,5,4,3,2,1,0,7);
    const __m256i right_mask=_mm256_set_epi32(0,7,6,5,4,3,2,1);
#pragma omp parallel for simd schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        int jx = jj * params.nx;
        int y_n = (jj + 1) % params.ny;
        int y_s = (jj != 0) ? (jj-1) : (params.ny - 1);
        int ynx = y_n*params.nx;
        int ysx = y_s*params.nx;
        for (int ii = 0; ii < (iteration/SIMDLEN+1)*SIMDLEN && ii < params.nx; ii+=SIMDLEN)
        {
            int pos= ii + jj*params.nx;
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */

            int x_e = (ii + SIMDLEN) % params.nx;
            int x_w = (ii != 0) ? (ii-1) : (params.nx - 1);

            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            int pos_set=pos/SIMDLEN;
            _mm256_store_ps(&(*cells)[pos_set].speed[0][0],_mm256_load_ps(&(*tmp_cells)[pos_set].speed[0][0])); /* central cell, no movement */
            int up_set=(ynx+ii)/SIMDLEN;
            int down_set=(ysx+ii)/SIMDLEN;
            speed_update(cells,tmp_cells,1,pos_set,pos_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update(cells,tmp_cells,3,pos_set,pos_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);

            speed_update(cells,tmp_cells,4,pos_set,up_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update(cells,tmp_cells,7,pos_set,up_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update(cells,tmp_cells,8,pos_set,up_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);

            speed_update(cells,tmp_cells,2,pos_set,down_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update(cells,tmp_cells,6,pos_set,down_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update(cells,tmp_cells,5,pos_set,down_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            /*left wall*/
            if(ii==0){
                const float cst1 =2.0/3.0;
                const float cst2 =1.0/6.0;
                const float cst3 =1.0/2.0;
                float local_density = ((*cells)[pos_set].speed[0][0]
                                       + (*cells)[pos_set].speed[2][0]
                                       + (*cells)[pos_set].speed[4][0]
                                       + 2.0 * ((*cells)[pos_set].speed[3][0]+(*cells)[pos_set].speed[6][0]+(*cells)[pos_set].speed[7][0])
                                      )/(1.0 - inlets[jj]);
                float local_inlet = local_density*inlets[jj];
                float index_cell = (*cells)[pos_set].speed[2][0]-(*cells)[pos_set].speed[4][0];

                (*cells)[pos_set].speed[1][0] = (*cells)[pos_set].speed[3][0]
                                                + cst1*local_inlet;

                (*cells)[pos_set].speed[5][0] = (*cells)[pos_set].speed[7][0]
                                                - cst3*(index_cell)
                                                + cst2*local_inlet;

                (*cells)[pos_set].speed[8][0] = (*cells)[pos_set].speed[6][0]
                                                + cst3*(index_cell)
                                                + cst2*local_inlet;
            }
            else if(ii==params.nx-SIMDLEN){
                (*cells)[pos_set].speed[0][SIMDLEN-1] = (*cells)[pos_set].speed[0][SIMDLEN-2];
                (*cells)[pos_set].speed[1][SIMDLEN-1] = (*cells)[pos_set].speed[1][SIMDLEN-2];
                (*cells)[pos_set].speed[2][SIMDLEN-1] = (*cells)[pos_set].speed[2][SIMDLEN-2];
                (*cells)[pos_set].speed[3][SIMDLEN-1] = (*cells)[pos_set].speed[3][SIMDLEN-2];
                (*cells)[pos_set].speed[4][SIMDLEN-1] = (*cells)[pos_set].speed[4][SIMDLEN-2];
                (*cells)[pos_set].speed[5][SIMDLEN-1] = (*cells)[pos_set].speed[5][SIMDLEN-2];
                (*cells)[pos_set].speed[6][SIMDLEN-1] = (*cells)[pos_set].speed[6][SIMDLEN-2];
                (*cells)[pos_set].speed[7][SIMDLEN-1] = (*cells)[pos_set].speed[7][SIMDLEN-2];
                (*cells)[pos_set].speed[8][SIMDLEN-1] = (*cells)[pos_set].speed[8][SIMDLEN-2];
            }
        }
    }
    return EXIT_SUCCESS;
}

/*
** Work with boundary conditions. The upper and lower boundaries use the rebound plane,
** the left border is the inlet of fixed speed, and
** the right border is the open outlet of the first-order approximation.
*/

/*data->tmp_cells*/
static inline void collision_obstacle_atom(const t_param params, t_speed_t** tmp_cells, __m256 data[NSPEEDS], __m256i obstacles, int set) {
    /* loop over the cells in the grid
    ** the collision step is called before
    ** the streaming step and so values of interest
    ** are in the scratch-space grid */
    const __m256 zero_vec=_mm256_setzero_ps();
    /*const __m256 one_vec=_mm256_set1_ps(1.f);*/
    __m256i one_int_vec=_mm256_set1_epi32(1);
    const __m256 half_vec=_mm256_set1_ps(0.5f);
    const __m256 three_vec=_mm256_set1_ps(3.f);
    /*const __m256 c_sq_vec=_mm256_set1_ps(c_sq);*/
    const __m256 w0_vec=_mm256_set1_ps(w0);
    const __m256 w1_vec=_mm256_set1_ps(w1);
    const __m256 w2_vec=_mm256_set1_ps(w2);



    __m256i obstacle_mask_inv=_mm256_xor_si256(obstacles,one_int_vec);
    if (!_mm256_testz_si256(obstacle_mask_inv,obstacle_mask_inv)){

        __m256 local_density_vec = _mm256_add_ps( _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(data[0],data[1]),_mm256_add_ps(data[2],data[3])),
                                                                _mm256_add_ps(_mm256_add_ps(data[4],data[5]),_mm256_add_ps(data[6],data[7]))),data[8]);

        __m256 u_x_vec =_mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(data[1],data[5]),_mm256_sub_ps(data[8],data[3])),_mm256_add_ps(data[6],data[7])),local_density_vec);
        __m256 u_y_vec =_mm256_div_ps(_mm256_sub_ps(_mm256_add_ps(_mm256_add_ps(data[2],data[5]),_mm256_sub_ps(data[6],data[4])),_mm256_add_ps(data[7],data[8])),local_density_vec);
        /* velocity squared */
        /* float u_sq = u_x * u_x + u_y * u_y; */
        __m256 u_sq_vec= _mm256_add_ps(_mm256_mul_ps(u_x_vec,u_x_vec),_mm256_mul_ps(u_y_vec,u_y_vec));
        __m256 u_sq_vec_3=_mm256_mul_ps(u_sq_vec,_mm256_set1_ps(-3.f));
        /* directional velocity components */


        __m256 u_vec[NSPEEDS];
        u_vec[0]=zero_vec;
        u_vec[1]= _mm256_mul_ps(u_x_vec,three_vec);
        u_vec[2]=_mm256_mul_ps(u_y_vec,three_vec);
        u_vec[3]=_mm256_sub_ps(zero_vec,u_vec[1]);
        u_vec[4]=_mm256_sub_ps(zero_vec,u_vec[2]);
        u_vec[5]=_mm256_add_ps(u_vec[1],u_vec[2]);
        u_vec[6]= _mm256_sub_ps(u_vec[2],u_vec[1]);
        u_vec[7]= _mm256_add_ps(u_vec[3],u_vec[4]);
        u_vec[8]= _mm256_sub_ps(u_vec[1],u_vec[2]);

        /* equilibrium densities */
        __m256 d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        /*u[0]=0, d_equ=w0*dens*(1+0+0-3/2 * u_sq)=w0*dens*(1-3/2 * u_sq)=w0*( dens*(-3/2 * u_sq)+dens )*/

        d_equ[0] = _mm256_mul_ps(w0_vec,_mm256_fmadd_ps(_mm256_mul_ps(u_sq_vec_3,half_vec),local_density_vec,local_density_vec));

        /*c_sq=1/3   (w1*dens) * (0.5*(3u*3u-3u_sq)+3u) + (w1*dens) = (w1*dens) * (9/2 * (u*u)- 3/2 *u_sq +3u+1)*/
        /* axis speeds: weight w1 */
        __m256 w1dens= _mm256_mul_ps(w1_vec,local_density_vec);
        d_equ[1] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[1],u_vec[1],u_sq_vec_3),half_vec,u_vec[1]),w1dens,w1dens);
        d_equ[2] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[2],u_vec[2],u_sq_vec_3),half_vec,u_vec[2]),w1dens,w1dens);
        d_equ[3] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[3],u_vec[3],u_sq_vec_3),half_vec,u_vec[3]),w1dens,w1dens);
        d_equ[4] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[4],u_vec[4],u_sq_vec_3),half_vec,u_vec[4]),w1dens,w1dens);

        /* diagonal speeds: weight w2 */
        __m256 w2dens= _mm256_mul_ps(w2_vec,local_density_vec);
        d_equ[5] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[5],u_vec[5],u_sq_vec_3),half_vec,u_vec[5]),w2dens,w2dens);
        d_equ[6] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[6],u_vec[6],u_sq_vec_3),half_vec,u_vec[6]),w2dens,w2dens);
        d_equ[7] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[7],u_vec[7],u_sq_vec_3),half_vec,u_vec[7]),w2dens,w2dens);
        d_equ[8] = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(u_vec[8],u_vec[8],u_sq_vec_3),half_vec,u_vec[8]),w2dens,w2dens);

        /* simd */
        /* printf("%f\n",(*cells)[pos].speeds[1]); */
        __m256 omega_vec=_mm256_set1_ps(params.omega);
        __m256 obstacle_mask=_mm256_castsi256_ps(_mm256_cmpeq_epi32(obstacle_mask_inv, one_int_vec));
        _mm256_store_ps(&(*tmp_cells)[set].speed[0][0],_mm256_blendv_ps(data[0],_mm256_add_ps(data[0],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[0],data[0]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[1][0],_mm256_blendv_ps(data[3],_mm256_add_ps(data[1],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[1],data[1]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[2][0],_mm256_blendv_ps(data[4],_mm256_add_ps(data[2],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[2],data[2]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[3][0],_mm256_blendv_ps(data[1],_mm256_add_ps(data[3],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[3],data[3]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[4][0],_mm256_blendv_ps(data[2],_mm256_add_ps(data[4],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[4],data[4]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[5][0],_mm256_blendv_ps(data[7],_mm256_add_ps(data[5],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[5],data[5]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[6][0],_mm256_blendv_ps(data[8],_mm256_add_ps(data[6],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[6],data[6]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[7][0],_mm256_blendv_ps(data[5],_mm256_add_ps(data[7],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[7],data[7]))),obstacle_mask));
        _mm256_store_ps(&(*tmp_cells)[set].speed[8][0],_mm256_blendv_ps(data[6],_mm256_add_ps(data[8],_mm256_mul_ps(omega_vec,_mm256_sub_ps(d_equ[8],data[8]))),obstacle_mask));

        /* relaxation step */
        /*for (int kk = 0; kk < NSPEEDS; kk++)
        {
          (*tmp_cells)[pos].speeds[kk] = (*cells)[pos].speeds[kk]+ params.omega * (d_equ[kk] - (*cells)[pos].speeds[kk]);
        }*/
    } else{
        _mm256_store_ps(&(*tmp_cells)[set].speed[0][0],data[0]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[1][0],data[3]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[2][0],data[4]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[3][0],data[1]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[4][0],data[2]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[5][0],data[7]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[6][0],data[8]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[7][0],data[5]);
        _mm256_store_ps(&(*tmp_cells)[set].speed[8][0],data[6]);
    }


}
/*tmp_cells->data*/
static inline void speed_update_atom(__m256 data[NSPEEDS],t_speed_t ** tmp_cells,int dir,int pos_set,int neighbour_set,int x_w, int jx,int ii,int ysx,int x_e,int ynx,const __m256i* left_mask,const __m256i* right_mask){
    int tmp_pos,set,ind;
    if(dir==1) { tmp_pos = x_w + jx;}
    else if (dir==2) tmp_pos=ii  + ysx;
    else if (dir==3) { tmp_pos = x_e + jx; }
    else if (dir==4) tmp_pos=ii  + ynx;
    else if (dir==5) { tmp_pos = x_w + ysx; }
    else if (dir==6) { tmp_pos = x_e + ysx; }
    else if (dir==7) { tmp_pos = x_e + ynx; }
    else if (dir==8) { tmp_pos = x_w + ynx; }

    int source_dir=dir;
    set=tmp_pos/SIMDLEN; ind=tmp_pos%SIMDLEN;
//  printf("pos_x: %d, pos_y:%d ,dir: %d ,temp_x: %d, temp_y:%d\n",pos_set*8%4096,pos_set*8/4096,dir,tmp_pos%4096,tmp_pos/4096);
    if(dir==2 || dir==4){
        if(ynx==0 && dir==4) { source_dir = 2;set=pos_set; }
        else if(jx==0 && dir==2) { source_dir = 4; set=pos_set; }

        data[dir]=_mm256_load_ps(&(*tmp_cells)[set].speed[source_dir][0]);
    }
    else if(dir==1 || dir==5 || dir==8){
        if(ynx==0 && dir==8) { source_dir = 6; neighbour_set=pos_set; data[dir]=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);}
        else if(jx==0 && dir==5) { source_dir = 7; neighbour_set=pos_set; data[dir]=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);}
        else{
            __m256 tmp=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);
            tmp[7]=(*tmp_cells)[set].speed[dir][ind];
            data[dir]=_mm256_permutevar8x32_ps(tmp,*left_mask);
        }
    }
    else{
        /*3 6 7*/
        if(ynx==0 && dir==7) { source_dir = 5;neighbour_set=pos_set; data[dir]=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);}
        else if(jx==0 && dir==6) { source_dir = 8; neighbour_set=pos_set; data[dir]=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);}
        else{
            __m256 tmp=_mm256_load_ps(&(*tmp_cells)[neighbour_set].speed[source_dir][0]);
            tmp[0]=(*tmp_cells)[set].speed[dir][ind];
            data[dir]=_mm256_permutevar8x32_ps(tmp,*right_mask);
        }
    }
}

/*
** Particles flow to the corresponding cell according to their speed direaction.
*/
/*tmp_cells->tmp_cells*/
int streaming_boundary_collision(int iteration, const t_param params, t_speed_t** cells, t_speed_t** tmp_cells, float* inlets,int* obstacles) {
    /* loop over _all_ cells */
//  printf("%f\n",(*cells)[0].speed[5][0]);
//  omp_set_num_threads(8);
    const __m256i left_mask=_mm256_set_epi32(6,5,4,3,2,1,0,7);
    const __m256i right_mask=_mm256_set_epi32(0,7,6,5,4,3,2,1);
#pragma omp parallel for simd schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        int jx = jj * params.nx;
        int y_n = (jj + 1) % params.ny;
        int y_s = (jj != 0) ? (jj - 1) : (params.ny - 1);
        int ynx = y_n*params.nx;
        int ysx = y_s*params.nx;
        for (int ii = 0; ii < (iteration/SIMDLEN+1)*SIMDLEN &&  ii <params.nx; ii+=SIMDLEN)
        {
            int pos= ii+jx;


            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */

            int x_e = (ii + SIMDLEN) % params.nx;
            int x_w = (ii != 0) ? (ii - 1) : (params.nx - 1);

            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */
            __m256 data[NSPEEDS];
            int pos_set=pos/SIMDLEN;
            data[0]=_mm256_load_ps(&(*tmp_cells)[pos_set].speed[0][0]); /* central cell, no movement */
            int up_set=(ynx+ii)/SIMDLEN;
            int down_set=(ysx+ii)/SIMDLEN;
            speed_update_atom(data,tmp_cells,1,pos_set,pos_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update_atom(data,tmp_cells,3,pos_set,pos_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);

            speed_update_atom(data,tmp_cells,4,pos_set,up_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update_atom(data,tmp_cells,7,pos_set,up_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update_atom(data,tmp_cells,8,pos_set,up_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);

            speed_update_atom(data,tmp_cells,2,pos_set,down_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update_atom(data,tmp_cells,6,pos_set,down_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            speed_update_atom(data,tmp_cells,5,pos_set,down_set,x_w,jx,ii,ysx,x_e,ynx,&left_mask,&right_mask);
            /*left wall*/
            if(ii==0){
                const float cst1 =2.0/3.0;
                const float cst2 =1.0/6.0;
                const float cst3 =1.0/2.0;
                float local_density = (  data[0][0]
                                         + data[2][0]
                                         + data[4][0]
                                         + 2.0 * (data[3][0]+data[6][0]+data[7][0])
                                      )/(1.0 - inlets[jj]);
                float local_inlet = local_density*inlets[jj];
                float index_cell = data[2][0]-data[4][0];

                data[1][0] = data[3][0]+ cst1*local_inlet;
                data[5][0] = data[7][0]- cst3*(index_cell)+ cst2*local_inlet;
                data[8][0] = data[6][0]+ cst3*(index_cell)+ cst2*local_inlet;
            }
            else if(ii==params.nx-SIMDLEN){
                data[0][SIMDLEN-1] = data[0][SIMDLEN-2];
                data[1][SIMDLEN-1] = data[1][SIMDLEN-2];
                data[2][SIMDLEN-1] = data[2][SIMDLEN-2];
                data[3][SIMDLEN-1] = data[3][SIMDLEN-2];
                data[4][SIMDLEN-1] = data[4][SIMDLEN-2];
                data[5][SIMDLEN-1] = data[5][SIMDLEN-2];
                data[6][SIMDLEN-1] = data[6][SIMDLEN-2];
                data[7][SIMDLEN-1] = data[7][SIMDLEN-2];
                data[8][SIMDLEN-1] = data[8][SIMDLEN-2];
            }
            collision_obstacle_atom(params,cells,data,_mm256_load_si256((__m256i *)&obstacles[pos]),pos_set);
        }
    }
    t_speed_t* switch_cell=*cells;
    *cells=*tmp_cells;
    *tmp_cells=switch_cell;
    return EXIT_SUCCESS;
}

