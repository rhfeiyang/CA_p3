#include "calc.h"

/* set inlets velocity there are two type inlets*/
int set_inlets(const t_param params, float* inlets) {
  for(int jj=0; jj <params.ny; jj++){
    if(!params.type)
      inlets[jj]=params.velocity; // homogeneous
    else
      inlets[jj]=params.velocity * 4.0 *((1-((float)jj)/params.ny)*((float)(jj+1))/params.ny); // parabolic
  }
  return EXIT_SUCCESS;
}

/* compute average velocity of whole grid, ignore grids with obstacles. */
float av_velocity(const t_param params, t_speed_t* cells, int* obstacles)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float  tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
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

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[kk].cells[pos];
        }

        /* x-component of velocity */
        float u_x = (   cells[1].cells[pos]
                      + cells[5].cells[pos]
                      + cells[8].cells[pos]
                      - (  cells[3].cells[pos]
                         + cells[6].cells[pos]
                         + cells[7].cells[pos]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (      cells[2].cells[pos]
                         + cells[5].cells[pos]
                         + cells[6].cells[pos]
                         -(cells[4].cells[pos]
                         + cells[7].cells[pos]
                         + cells[8].cells[pos]))
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
float calc_reynolds(const t_param params, t_speed_t* cells, int* obstacles)
{
  return av_velocity(params, cells, obstacles) * (float)(params.ny) / params.viscosity;
}