/*
 * Droplet - Multiple Particle Collision
 * - Immersed Boundary: imposed momentum
 * - All-mach formulation, conservative approach
 *
 * Compile with:
 * 
 *      3D with MPI: CC99='mpicc -std=c99' qcc -Wall -O2 -grid=octree -D_MPI=1 droplet.c -o droplet -lm -L$BASILISK/gl -lglutils -lfb_glx -lGLU -lGLEW -lGL -lX11
 *      
 *      2D with MPI: CC99='mpicc -std=c99' qcc -Wall -O2 -D_MPI=1 droplet.c -o droplet -lm -L$BASILISK/gl -lglutils -lfb_glx -lGLU -lGLEW -lGL -lX11
 * 
 *      2D without MPI: qcc -Wall -O2 droplet.c -o droplet -lm
 * 
 * Warning: mkdir ./results
 * Run with:
 * 
 *      with MPI: mpirun -np 2 ./droplet
 *      without MPI: ./droplet
 * 
 * Optional:
 * 
 * qcc -source droplet.c
 * mpicc -O2 -Wall -std=c99 -D_MPI=1 _droplet.c -o droplet -lm
 * mpirun -np 2 ./droplet
 * 
 * 
 * Output:
 * You may want to build a gif animation further
 * $ convert -delay 10 -loop 0 *.png animation.gif
 * 
 */

#include <stdlib.h>
#include <time.h>

#include "momentum.h"
#include "tension.h"
// #include "view.h"
#include "droplet.h"

#if dimension == 3
#include "lambda2.h"
#endif

#define cf


// Droplet diameter
#define Dd 1.e-3

// Gravity acceleration
#define GRAVITY 9.81

// Fluid flow properties. Liquid: water, Gas: air
#define RHO_L 1000.
#define RHO_G 1.
#define MU_L 1.e-3
#define MU_G 1.e-6

// Solid density (any)
#define RHO_S 500.

// Randomly chosen dimensionless parameters
// Re [10³ - 10⁴]
int Re = 0; // RHO_L*Vd*Dd/MU_L
// We [10² - 10³]
int We = 0; // RHO_L*Vd*Vd*Dd/SIGMA 

// Initial droplet velocity based on randomly Re
// Vd = Re * MU_L / ( RHO_L * Dd ) 
double Vd = 0.; 

// Water-air surface tension coefficient based on randomly We
// SIGMA = RHO_L * Vd * Vd * Dd / We
double SIGMA = 0.;

// Solid-Gas volume fraction
double Csg = 0.;

// Liquid-Solid volume fraction
double Cls = 0.;

// Liquid-Solid area ratio;
double Als = 0.;

#define PRESSURE 101000

#define H 18*Dd
#define h 8*Dd

#if dimension == 2
  #define particle_zone_volume h*h
#else
  #define particle_zone_volume h*h*h
#endif


// Particles
scalar particle[];
scalar particles[];

// Randomly chosen particle parameters
// number_of_particles [1 - 4]
int number_of_particles = 0; 
// Dp [Dd/2 - 2*Dd]
double Dp[MAX_PARTICLES] = {0.};
// Inside particles zone without overlap
double Xp[MAX_PARTICLES] = {0.};
double Yp[MAX_PARTICLES] = {0.};
#if dimension == 3
double Zp[MAX_PARTICLES] = {0.};
#endif


// Droplet
#define Xd 0.
#define Yd (H - 3*Dd)
#define Ud 0.
#if dimension == 3
#define Zd 0.
#define Wd 0.
#endif

#if dimension == 2
const double theoretical_initial_droplet_area = pi*Dd; 
#else
const double theoretical_initial_droplet_area = pi*Dd*Dd;
#endif

#if dimension == 2
const double theoretical_initial_droplet_volume = pi*Dd*Dd/4; 
#else
const double theoretical_initial_droplet_volume = (4/3)*pi*pow(Dd/2; 3);
#endif


// Dataset Outputs
double mean_x1 = 0.;
double var_x1 = 0.;
double mean_x2 = 0;
double var_x2 = 0;
// ...with auxiliaries
double sum_w = 0.;
double sum_wx1 = 0.;
double sum_wx2 = 0.;
// ... and data at last time step (last sample)
double old_sum_w = 0.;
double old_mean_x1 = 0.;
double old_var_x1 = 0.;
double old_mean_x2 = 0;
double old_var_x2 = 0;


// Refinement level (level, n): (5, 32) (6, 64) (7, 128) (8, 256) (9, 512) (10, 1024)
int LEVEL_MIN = 5;
int LEVEL_RFN = 8;
int LEVEL_MAX = 9;

// Checking time and simulation duration
const double dimensionless_duration = 9.5;


int current_run;
int start_run_at = 1;


FILE *dataset;
FILE *simulationLog;

int main() {
  
  L0 = H;

  #if dimension == 2    
  origin(-0.5*H, 0.);
  #else
  origin(-0.5*H, 0., -0.5*H);
  #endif  

  foreach_dimension()
    periodic (right);  
    
  rho1 = RHO_L;
  rho2 = RHO_G;    

  mu1 = MU_L;
  mu2 = MU_G;
  
  // Decreasing the tolerance on the Poisson solve improves the results 
  TOLERANCE = 1e-4;
        
  // Initial uniform mesh size 
  N = 1 << LEVEL_MIN;
  init_grid(N);
  
  // Randomly Setup 
  int number_of_runs = 1000;
    
  // Rand's seed
  // Optional: srand(time(NULL))
  srand(0);
  
  printf("srand\n");
  
  dataset = fopen ("./dataset/dataset.csv", "w");
  write_dataset_header (dataset); 
  printf("write_dataset_header\n");
  
  for (int nr=start_run_at; nr<start_run_at+number_of_runs; nr++) {
    
    current_run = nr;
    printf("current_run=%d\n",current_run);
    
    // Re [10³ - 10⁴] 
    Re = (double)(1000 + rand()%9000);
    
    printf("Re=%d\n",Re);
    
    Vd = Re * MU_L / ( RHO_L * Dd ); 
    
    printf("Vd=%g\n",Vd);
    
    // We [10² - 10³] 
    We = (double)(100 + rand()%900);
    
    printf("We=%d\n",We);
    
    SIGMA = RHO_L * Vd * Vd * Dd / We;    
    f.sigma = SIGMA;
    
    printf("SIGMA=%g\n",SIGMA);
    
    // number_of_particles [0 - 4]
    number_of_particles = 1 + rand()%4;
    
    printf("np=%d\n",number_of_particles);
        
    double theoretical_particles_area = 0.;
    double theoretical_particles_volume = 0.;
        
    for (int np=0; np<number_of_particles; np++) {
      
      Dp[np] = 0.5*Dd + 1.5*Dd*(double)rand()/RAND_MAX;      
      printf("Dp[%d]=%g\n",np,Dp[np]);
      
      theoretical_particles_area += pi*Dp[np];
      theoretical_particles_volume += pi*Dp[np]*Dp[np]/4; 
     
      int overlap = 1; 
      while (overlap == 1) {        
        Xp[np] = -4*Dd + 0.5*Dp[np] + (8*Dd - Dp[np])*(double)rand()/RAND_MAX;
        Yp[np] = 6*Dd + 0.5*Dp[np] + (8*Dd - Dp[np])*(double)rand()/RAND_MAX;
        overlap = check_overlap (np, Dp, Xp, Yp);
        printf("overlap\n");
      }        
      printf("Xp[%d]=%g\n",np,Xp[np]);
      printf("Yp[%d]=%g\n",np,Yp[np]);
              
    }
    for (int np=number_of_particles; np<MAX_PARTICLES; np++) {
      
      Dp[np] = 0.;      
      printf("Dp[%d]=%g\n",np,Dp[np]);
      
      Xp[np] = 0.;
      Yp[np] = 0.;
      printf("Xp[%d]=%g\n",np,Xp[np]);
      printf("Yp[%d]=%g\n",np,Yp[np]);      
      
    }
        
    printf("theoretical_particles_area=%g\n",theoretical_particles_area);
    printf("theoretical_particles_volume=%g\n",theoretical_particles_volume);
           
    Csg = theoretical_particles_volume/particle_zone_volume;
    printf("Csg=%g\n",Csg);

    Cls = theoretical_initial_droplet_volume/theoretical_particles_volume;
    printf("Cls=%g\n",Cls);
    
    Als = theoretical_initial_droplet_area/theoretical_particles_area;
    printf("Als=%g\n",Als);
        
    write_dataset_input (dataset, nr, Re, We, number_of_particles, Dp, Csg, Cls, Als, Xp, Yp);
    printf("write_dataset_input\n");

    
    // Open Event Files
    simulationLog = fopen ("./logs/simulationLog.csv", "a");
    
    
    run(); 
    
  }
  
  fclose(dataset);
  fclose(simulationLog);
  
}


// Periodic boundary conditions at main () 
// Check compatibility at http://basilisk.fr/src/COMPATIBILITY

// #ifdef 0
// q.n[bottom] = q.n[] < 0 ? neumann(0) : 0;
// p[bottom] = dirichlet(PRESSURE);
// 
// q.n[top] = neumann(0);
// p[top] = dirichlet(PRESSURE);
// 
// q.n[right] = neumann(0);
// p[right] = dirichlet(PRESSURE);
// 
// q.n[left] = neumann(0);
// p[left] = dirichlet(PRESSURE);
// #endif


event init(i = 0) {
  
  if (!restore (file = "dump")) {  
    
    coord droplet_velocity = {Ud, -Vd};   
    refine ( ( ( fabs(x)<4*Dd && y>6*Dd && y<14*Dd ) ||  ( sq(x-Xd)+sq(y-Yd)<sq(1.2*Dd/2) && sq(x-Xd)+sq(y-Yd)>sq(0.8*Dd/2) ) ) && level<LEVEL_RFN );         
    fraction ( f, sq(0.5*Dd) - sq(x-Xd) - sq(y-Yd) );
      
    for (int np=0; np<number_of_particles; np++) {
      fraction ( particle, sq(0.5*Dp[np])-sq(x-Xp[np])-sq(y-Yp[np]) );  
      foreach()
        particles[] = particles[] + particle[];
    }
    
    foreach() 
      foreach_dimension()      
        q.x[] = f[]*rho1*droplet_velocity.x;
    boundary ((scalar *){q}); 
    
    #ifndef cf
      foreach()
        cf[] = f[];
    #endif         
    
  }  
  
}


event initial_field (i=0) {
  
  char ppm_name_vof[80];
  sprintf(ppm_name_vof, "./images/initial-vof-run-%d.ppm", current_run);
        
  char ppm_name_level[80]; 
  sprintf(ppm_name_level, "./images/initial-level.ppm");
    
  scalar ff[], ll[], m[];
  vector u[];
  foreach() {
    ff[] = f[] < 1e-4 ? 0 : f[] > 1. - 1e-4 ? 1. : f[];
    ll[] = level;
    m[] = 0.5 - particles[];
    foreach_dimension()
      u.x[] = q.x[]/rho[];
  }
  boundary({ff, ll, m});
  boundary((scalar *){u});
  
  output_ppm(ff, file=ppm_name_vof, mask=m, n=1<<LEVEL_MAX);
  
  if (current_run == 1)
    output_ppm(ll, file=ppm_name_level, max=LEVEL_MAX, n=1<<LEVEL_MAX);
   
}


event immersed_boundary (i++) {
    
  coord particle_velocity = {0., 0.};  
  
  foreach()
    foreach_dimension()
      q.x[] = particles[]*RHO_S*particle_velocity.x + (1. - particles[])*q.x[];
  boundary ((scalar *){q});
  
}


event acceleration (i++) {
  
  face vector av = a;
  foreach_face(y)
    av.y[] -= GRAVITY;  
  
}


event adapt (i++) {
  
  vector u[];
  foreach()
    foreach_dimension()
      u.x[] = q.x[]/rho[];
  boundary((scalar *){u});

  adapt_wavelet({f, particles, u}, (double[]){1.e-3, 1.e-3, 1.e-2, 1.e-2}, LEVEL_MAX);
  
}


double wetted_area(scalar f, scalar particles) {

  // General orthogonal coordinates
  // http://basilisk.fr/src/README
  
  double wetted = 0.;
  
  foreach() {
    
    if (particles[] > 0.5) {
      
      foreach_dimension() {        
        if (particles[1] < 0.5 && f[1] > 0.5 )
          wetted += fm.x[1]*Delta;
        if (particles[-1] < 0.5 && f[-1] > 0.5 )
          wetted += fm.x[]*Delta;         
      }     
      
    }   
    
  }  
  
  return wetted;  
  
}


event compute_outputs (i=0; i++) {
  
  double wa = wetted_area(f, particles);
  double la = interface_area(f);
  
  // Incremental calculation of weighted mean and variance. Tony Finch, University of Cambridge. February 2009.
  
  sum_w += dt;
  sum_wx1 += dt*wa;
  sum_wx2 += dt*la;
  
  mean_x1 = sum_wx1/sum_w;
  mean_x2 = sum_wx2/sum_w;
  
  if (i == 0) {
    var_x1 = 0.;
    var_x2 = 0.;
  }
  else {
    var_x1 = ((sum_w-dt)/old_sum_w)*old_var_x1 + dt*(wa-old_mean_x1)*(wa-mean_x1);
    var_x2 = ((sum_w-dt)/old_sum_w)*old_var_x2 + dt*(la-old_mean_x2)*(la-mean_x2);  
  }
  
  old_mean_x1 = mean_x1;
  old_mean_x2 = mean_x2;
  old_var_x1 = var_x1;
  old_var_x2 = var_x2;
  old_sum_w = sum_w;
    
  if (t > dimensionless_duration*Dd/Vd) {   
    
    printf ("\nt > dimensionless_duration*Dd/Vd = %e\n", dimensionless_duration*Dd/Vd);     
  
    write_dataset_output(dataset, mean_x1, sqrt(var_x1/sum_w), mean_x2, sqrt(var_x2/sum_w)); 
    printf ("write_dataset_output\n");
    
    return 1; 
    
  }
  
}


event console_log (i+=10) {
  if (i%100 == 0)
    fprintf (ferr, "i t t* dt\n");
  fprintf (ferr, "%d %e %e %g\n", i, t, t*Vd/Dd, dt);
}


event simulation_log (i+=100; i<100000) {
     
  if (i == 0)
    fprintf (simulationLog, "run, t, dt, mgp.i, mgu.i, grid->tn, perf.t, perf.speed\n");  
  fprintf (simulationLog, "%d, %g, %g, %d, %d, %ld, %g, %g\n", current_run, t, dt, mgp.i, mgu.i, grid->tn, perf.t, perf.speed);  
  fflush (simulationLog);  
  
}


// event fields (i=200; i+=200) {
//   
//   char ppm_name_vof[80];
//   sprintf(ppm_name_vof, "./images/vof-run-%d-t-%.6g.ppm", current_run, t*Vd/Dd);
//         
//   char ppm_name_level[80]; 
//   sprintf(ppm_name_level, "./images/level-run-%d-t-%.6g.ppm", current_run, t*Vd/Dd);
//     
//   char ppm_name_vort[80]; 
//   sprintf(ppm_name_vort, "./images/vort-run-%d-t-%.6g.ppm", current_run, t*Vd/Dd);
//   
//   scalar ff[], ll[], m[], omega[];
//   vector u[];
//   foreach() {
//     ff[] = f[] < 1e-4 ? 0 : f[] > 1. - 1e-4 ? 1. : f[];
//     ll[] = level;
//     m[] = 0.5 - particles[];
//     foreach_dimension()
//       u.x[] = q.x[]/rho[];
//   }
//   boundary({ff, ll, m});
//   boundary((scalar *){u});
//   
//   vorticity (u, omega);
//   
//   output_ppm(ff, file=ppm_name_vof, mask=m, n=1<<LEVEL_MAX);
//   output_ppm(ll, file=ppm_name_level, max=LEVEL_MAX, n=1<<LEVEL_MAX);
//   output_ppm(omega, file=ppm_name_vort, mask=m, n=1<<LEVEL_MAX);
//    
// }



   
