#ifndef TYPES_H
#define TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
//#include <sys/resource.h>

#define NSPEEDS         9
#define NUM_THREADS     28
#define SIMDLEN        8
typedef struct
{
  int    ny;            /* no. of cells in y-direction */
  int    nx;            /* no. of cells in x-direction */
  float  omega;         /* relaxation parameter */
  int    maxIters;      /* no. of iterations */
  float  density;       /* density per cell */
  float  viscosity;     /* kinematic viscosity of fluid */
  float  velocity;      /* inlet velocity */
  int    type;          /* inlet type */
  
} t_param;

/* struct to hold the distribution of different speeds */
/*typedef struct
{
  float speeds[NSPEEDS];
} t_speed;*/

typedef struct{
    /* (n/8) *9 * 8 */
    float speed[NSPEEDS][SIMDLEN];
} t_speed_t;

/*typedef struct
{
    *//* (n/8) * 9 * 8 *//*
    t_speed_block* blocks;
} t_speed_t;*/

#endif