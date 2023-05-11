#ifndef TYPES_H
#define TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NSPEEDS         9
#define NUM_THREADS     28

typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    type;          /* inlet type */
  int    maxIters;      /* no. of iterations */
  float  omega;         /* relaxation parameter */
  float  density;       /* density per cell */
  float  viscosity;     /* kinematic viscosity of fluid */
  float  velocity;      /* inlet velocity */
  
} t_param;

/* struct to hold the distribution of different speeds */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

#endif