#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float* inlets) {
#pragma omp parallel for schedule(static)
    for(int jj=0; jj <params.ny; jj++){
        if(!params.type)
            inlets[jj]=params.velocity; // homogeneous
        else
            inlets[jj]=params.velocity * 4.0 *((1-((float)jj)/params.ny)*((float)(jj+1))/params.ny); // parabolic
    }
    return EXIT_SUCCESS;
}

/* compute average velocity of whole grid, ignore grids with obstacles. */
float av_velocity(const t_param params, t_speed_t** cells, int* obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float  tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    /* loop over all non-blocked cells */
#pragma omp parallel for schedule(static)
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int pos = ii + jj*params.nx;
            /* ignore occupied cells */
            if (!obstacles[pos])
            {
                /* local density total */
                float local_density = 0.f;
                const int set=pos/SIMDLEN;
                int ind=pos%SIMDLEN;
                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    local_density += (*cells)[set].speed[kk][ind];
                }

                /* x-component of velocity */
                float u_x = (   (*cells)[set].speed[1][ind]
                                + (*cells)[set].speed[5][ind]
                                + (*cells)[set].speed[8][ind]
                                - (  (*cells)[set].speed[3][ind]
                                     + (*cells)[set].speed[6][ind]
                                     + (*cells)[set].speed[7][ind]))
                            / local_density;
                /* compute y velocity component */
                float u_y = (      (*cells)[set].speed[2][ind]
                                   + (*cells)[set].speed[5][ind]
                                   + (*cells)[set].speed[6][ind]
                                   -((*cells)[set].speed[4][ind]
                                     + (*cells)[set].speed[7][ind]
                                     + (*cells)[set].speed[8][ind]))
                            / local_density;

                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}

/* calculate reynold number */
float calc_reynolds(const t_param params, t_speed_t** cells, int* obstacles)
{
    return av_velocity(params, cells, obstacles) * (float)(params.ny) / params.viscosity;
}
