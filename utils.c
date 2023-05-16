#include "types.h"

/* utility functions */
void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed_t** cells_ptr, t_speed_t** tmp_cells_ptr,
               int** obstacles_ptr, float** inlets_ptr)
{
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter nx */
    retval = fscanf(fp, "nx: %d\n", &(params->nx));
    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

    /* read in the parameter ny */
    retval = fscanf(fp, "ny: %d\n", &(params->ny));
    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

    /* read in the parameter maxIters */
    retval = fscanf(fp, "iters: %d\n", &(params->maxIters));
    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

    /* read in the parameter density */
    retval = fscanf(fp, "density: %f\n", &(params->density));
    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

    /* read in the parameter viscosity */
    retval = fscanf(fp, "viscosity: %f\n", &(params->viscosity));
    if (retval != 1) die("could not read param file: viscosity", __LINE__, __FILE__);

    /* read in the parameter velocity */
    retval = fscanf(fp, "velocity: %f\n", &(params->velocity));
    if (retval != 1) die("could not read param file: velocity", __LINE__, __FILE__);

    /* read in the parameter type */
    retval = fscanf(fp, "type: %d\n", &(params->type));
    if (retval != 1) die("could not read param file: type", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    /* calculation of relaxtion parameter */
    params->omega=1./(3.*params->viscosity+0.5);

    /* Check the calculation stability */
    if(params->velocity>0.2)
        printf("Warning: There maybe computational instability due to compressibility.\n");
    if((2-params->omega) < 0.15)
        printf("Warning: Possible divergence of results due to relaxation time.\n");

    /* Allocate memory. */

    /* main grid */
    /**cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);*/

    /* 'helper' grid, used as scratch space */
    /**tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));
    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);*/
    const int set_num=(params->ny * params->nx)/SIMDLEN;

    *cells_ptr= (t_speed_t*)aligned_alloc(32, sizeof(t_speed_t) * set_num);
    *tmp_cells_ptr= (t_speed_t*)aligned_alloc(32, sizeof(t_speed_t) * set_num);
    if(*cells_ptr == NULL || *tmp_cells_ptr==NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);


    /* the map of obstacles */
    /**obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));*/
    *obstacles_ptr= (int*)aligned_alloc(32, sizeof(int) * (params->ny * params->nx));
    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density      / 9.f;
    float w2 = params->density      / 36.f;

    /*for (int jj = 0; jj < params->ny; jj++)
    {
      for (int ii = 0; ii < params->nx; ii++)
      {
        *//* centre *//*
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      *//* axis directions *//*
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      *//* diagonals *//*
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }*/
#pragma omp parallel for schedule(static)
    for(int set=0;set<set_num;set++) {
        for (int i = 0; i < NSPEEDS; i++) {
            float w;
            if (i == 0) w = w0;
            else if (i > 0 && i < 5) w = w1;
            else w = w2;
            for (int j = 0; j < SIMDLEN; j++) {
                (*cells_ptr)[set].speed[i][j] = w;
            }
        }
    }
    /* first set all cells in obstacle array to zero */
    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            (*obstacles_ptr)[ii + jj*params->nx] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* Center the obstacle on the y-axis. */
        yy = yy + params->ny/2;

        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        (*obstacles_ptr)[xx + yy*params->nx] = blocked;
    }

    /* and close the file */
    fclose(fp);

    /* allocate space to hold the velocity of the cells at the inlet. */
    /**inlets_ptr = (float*)malloc(sizeof(float) * params->ny);*/
    *inlets_ptr= (float*)aligned_alloc(32, sizeof(float) * params->ny);
    return EXIT_SUCCESS;
}

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed_t** cells_ptr, t_speed_t** tmp_cells_ptr,
             int** obstacles_ptr, float** inlets)
{
    /*
    ** free up allocated memory
    */
    /*free(*cells_ptr);
    *cells_ptr = NULL;

    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;*/
    free(*cells_ptr);
    free(*tmp_cells_ptr);


    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*inlets);
    *inlets = NULL;

    return EXIT_SUCCESS;
}


/* write state of current grid */
int write_state(char* filename, const t_param params, t_speed_t** cells, int* obstacles)
{
    FILE* fp;                    /* file pointer */
    float local_density;         /* per grid cell sum of densities */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(filename, "w");

    if (fp == NULL)
    {
        printf("%s\n",filename);
        die("could not open file output file", __LINE__, __FILE__);
    }

    /* loop on grid to calculate the velocity of each cell */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            const int pos=ii + jj*params.nx;
            if (obstacles[pos])
            { /* an obstacle cell */
                u = -0.05f;
            }
            else
            { /* no obstacle */
                const int set=pos/SIMDLEN;
                int ind=pos%SIMDLEN;
                local_density = 0.f;

                for (int kk = 0; kk < NSPEEDS; kk++)
                {
                    /*local_density += (*cells)[pos].speeds[kk];*/
                    local_density += (*cells)[set].speed[kk][ind];
                }

                /* compute x velocity component */
                /*u_x = ((*cells)[pos].speeds[1]
                       + (*cells)[pos].speeds[5]
                       + (*cells)[pos].speeds[8]
                       - ((*cells)[pos].speeds[3]
                          + (*cells)[pos].speeds[6]
                          + (*cells)[pos].speeds[7]))
                      / local_density;*/
                u_x = ((*cells)[set].speed[1][ind]
                       + (*cells)[set].speed[5][ind]
                       + (*cells)[set].speed[8][ind]
                       - ((*cells)[set].speed[3][ind]
                          + (*cells)[set].speed[6][ind]
                          + (*cells)[set].speed[7][ind]))
                      / local_density;
                /* compute y velocity component */
                u_y = ((*cells)[set].speed[2][ind]
                       + (*cells)[set].speed[5][ind]
                       + (*cells)[set].speed[6][ind]
                       - ((*cells)[set].speed[4][ind]
                          + (*cells)[set].speed[7][ind]
                          + (*cells)[set].speed[8][ind]))
                      / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E\n", ii, jj, u);
        }
    }

    /* close file */
    fclose(fp);

    return EXIT_SUCCESS;
}
