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

// For MPI examples http://basilisk.fr/src/examples/atomisation.c

#include <stdlib.h>
#include <time.h>

#include "momentum.h"
#include "tension.h"
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
#elif dimension == 3
  #define particle_zone_volume h*h*h
#endif


// Particles
// scalar particle[];
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
#elif dimension == 3
const double theoretical_initial_droplet_area = pi*Dd*Dd;
#endif

#if dimension == 2
const double theoretical_initial_droplet_volume = pi*Dd*Dd/4; 
#elif dimension == 3
const double theoretical_initial_droplet_volume = (4/3)*pi*(Dd/2)*(Dd/2)*(Dd/2);
#endif


// Dataset Outputs: x1 and x2 are output variables
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
#define LEVEL_MIN 5
#define LEVEL_RFN 8
#define LEVEL_MAX 9

// Checking time and simulation duration
const double dimensionless_duration = 9.5;


int current_run;
int start_run_at = 1;



FILE *dataset;


    
int main() {
    
  L0 = H;

  #if dimension == 2    
  origin(-0.5*H, 0.);
  #elif dimension == 3
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
  int number_of_runs = 10;
    
  // Rand's seed
  // Optional: srand(time(NULL))
  srand(0);
  
  printf("srand\n");
  
  #if _OPENMP || _MPI
    if (pid() == 0) {
      dataset = fopen ("./dataset/dataset.csv", "w");
      write_dataset_header (dataset); 
      printf("\nwrite_dataset_header pid:%d", pid());
    }
  #else
    dataset = fopen ("./dataset/dataset.csv", "w");
    write_dataset_header (dataset); 
    printf("write_dataset_header\n");
  #endif
  
  
  for (int nr=start_run_at; nr<start_run_at+number_of_runs; nr++) {
    
    current_run = nr;
    printf("\ncurrent_run=%d pid:%d", current_run, pid());
    
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
      
      #if dimension == 2
        theoretical_particles_area += pi*Dp[np];
        theoretical_particles_volume += pi*Dp[np]*Dp[np]/4;
      #elif dimension == 3
        theoretical_particles_area += pi*Dp[np]*Dp[np];
        theoretical_particles_volume += (4/3)*pi*pow(Dp[np]/2, 3);
      #endif
	
      int overlap = 1; 
      while (overlap == 1) {        
        Xp[np] = -4*Dd + 0.5*Dp[np] + (8*Dd - Dp[np])*(double)rand()/RAND_MAX;
        Yp[np] = 6*Dd + 0.5*Dp[np] + (8*Dd - Dp[np])*(double)rand()/RAND_MAX;
        #if dimension == 2
          overlap = check_overlap (np, Dp, Xp, Yp);
        #elif dimension == 3
          Zp[np] = -4*Dd + 0.5*Dp[np] + (8*Dd - Dp[np])*(double)rand()/RAND_MAX; 
          overlap = check_overlap (np, Dp, Xp, Yp, Zp);
        #endif  
        printf("overlap pid:%d\n", pid());
      }        
      printf("Xp[%d]=%g pid:%d\n", np, Xp[np], pid());
      printf("Yp[%d]=%g pid:%d\n", np, Yp[np], pid());
      #if dimension == 3
        printf("Zp[%d]=%g pid:%d\n",np, Zp[np], pid());
      #endif
              
    }
    for (int np=number_of_particles; np<MAX_PARTICLES; np++) {
      
      Dp[np] = 0.;      
      printf("Dp[%d]=%g pid:%d\n", np, Dp[np], pid());
      
      Xp[np] = 0.;
      Yp[np] = 0.;
      printf("Xp[%d]=%g pid:%d\n", np, Xp[np], pid());
      printf("Yp[%d]=%g pid:%d\n", np, Yp[np], pid()); 
      #if dimension == 3
        Zp[np] = 0.;
        printf("Zp[%d]=%g pid:%d\n", np, Zp[np], pid());
      #endif           
      
    }
        
    printf("theoretical_particles_area=%g pid:%d\n",theoretical_particles_area, pid());
    printf("theoretical_particles_volume=%g pid:%d\n",theoretical_particles_volume, pid());
           
    Csg = theoretical_particles_volume/particle_zone_volume;
    printf("Csg=%g\n",Csg);

    Cls = theoretical_initial_droplet_volume/theoretical_particles_volume;
    printf("Cls=%g\n",Cls);
    
    Als = theoretical_initial_droplet_area/theoretical_particles_area;
    printf("Als=%g\n",Als);
            
    #if _OPENMP || _MPI
      if (pid() == 0) {
        #if dimension == 2
          write_dataset_input (dataset, nr, Re, We, number_of_particles, Dp, Csg, Cls, Als, Xp, Yp);
        #elif dimension == 3
          write_dataset_input (dataset, nr, Re, We, number_of_particles, Dp, Csg, Cls, Als, Xp, Yp, Zp);
        #endif
        printf("write_dataset_input pid:%d\n", pid());
      }
    #else
      #if dimension == 2
        write_dataset_input (dataset, nr, Re, We, number_of_particles, Dp, Csg, Cls, Als, Xp, Yp);
      #elif dimension == 3
        write_dataset_input (dataset, nr, Re, We, number_of_particles, Dp, Csg, Cls, Als, Xp, Yp, Zp);
      #endif
      printf("write_dataset_input pid:%d\n", pid());
    #endif
   
   
    run(); 
    
  }
  
  
}





event init (i = 0) {
  
  printf("\nStart init(i = 0) pid:%d", pid());
  
  //if (!restore (file = "dump")) {  
    
    #if dimension == 2
      coord droplet_velocity = {Ud, -Vd}; 
      refine ( ( ( fabs(x)<4*Dd && y>6*Dd && y<14*Dd ) ||  ( sq(x-Xd)+sq(y-Yd)<sq(1.2*Dd/2) && sq(x-Xd)+sq(y-Yd)>sq(0.8*Dd/2) ) ) && level<LEVEL_RFN );         
      fraction ( f, sq(0.5*Dd) - sq(x-Xd) - sq(y-Yd) );
    #elif dimension == 3
      coord droplet_velocity = {Ud, -Vd, Wd}; 
      refine ( ( ( fabs(x)<4*Dd && fabs(z)<4*Dd && y>6*Dd && y<14*Dd ) ||  ( sq(x-Xd)+sq(y-Yd)+sq(z-Zd)<sq(1.2*Dd/2) && sq(x-Xd)+sq(y-Yd)+sq(z-Zd)>sq(0.8*Dd/2) ) ) && level<LEVEL_RFN );         
      fraction ( f, sq(0.5*Dd)-sq(x-Xd)-sq(y-Yd)-sq(z-Zd) ); 
    #endif  

    #if dimension == 2
      fraction ( particles, max(
                                 (Dp[0]>0 ? sq(0.5*Dp[0])-sq(x-Xp[0])-sq(y-Yp[0]) : -1),
                                 max( 
                                      (Dp[1]>0 ? sq(0.5*Dp[1])-sq(x-Xp[1])-sq(y-Yp[1]) : -1),
                                      max( 
                                           (Dp[2]>0 ? sq(0.5*Dp[2])-sq(x-Xp[2])-sq(y-Yp[2]) : -1),
                                           (Dp[3]>0 ? sq(0.5*Dp[3])-sq(x-Xp[3])-sq(y-Yp[3]) : -1) ) ) ) );
    #elif dimension == 3      
      fraction ( particles, max(
                                 (Dp[0]>0 ? sq(0.5*Dp[0])-sq(x-Xp[0])-sq(y-Yp[0])-sq(z) : -1),
                                 max( 
                                      (Dp[1]>0 ? sq(0.5*Dp[1])-sq(x-Xp[1])-sq(y-Yp[1])-sq(z) : -1),
                                      max( 
                                           (Dp[2]>0 ? sq(0.5*Dp[2])-sq(x-Xp[2])-sq(y-Yp[2])-sq(z-Zp[2]) : -1),
                                           (Dp[3]>0 ? sq(0.5*Dp[3])-sq(x-Xp[3])-sq(y-Yp[3])-sq(z-Zp[3]) : -1) ) ) ) );
                        
    #endif
      
    foreach() 
      foreach_dimension()      
        q.x[] = f[]*rho1*droplet_velocity.x;
    boundary ((scalar *){q}); 
    
    #ifndef cf
      foreach()
        cf[] = f[];
    #endif         
    
  //}  
      
  printf("\nEnd init(i = 0) pid:%d", pid());      
        
}


event initial_field (i = 0) {
  
  printf("\nStart initial_field (i = 0) pid:%d", pid());
    
  char ppm_name_vof[80];
  sprintf(ppm_name_vof, "./images/initial-vof-run-%d-pid-%d.ppm", current_run, pid());
  
  char ppm_name_part[80];
  sprintf(ppm_name_part, "./images/initial-part-run-%d-pid-%d.ppm", current_run, pid());
        
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
  output_ppm(particles, file=ppm_name_part, n=1<<LEVEL_MAX);
  
  if (current_run == 1) {
    
    output_ppm(ll, file=ppm_name_level, max=LEVEL_MAX, n=1<<LEVEL_MAX);
  
    #if _OPENMP || _MPI
      char ppm_name_pid[80];
      sprintf(ppm_name_pid, "./images/initial-pid.ppm"); 
      scalar pid[];
      foreach()
        pid[] = tid();
      double tmax = npe() - 1;
      output_ppm (pid, file=ppm_name_pid, max=tmax, n=1<<LEVEL_MAX);
    #endif
      
  }
  
  printf("\nEnd initial_field (i = 0) pid:%d", pid());
   
}


event immersed_boundary (i++) {
  
  printf("\nStart immersed_boundary (i++) pid:%d", pid());
    
  #if dimension == 2
    coord particle_velocity = {0., 0.};  
  #elif dimension == 3  
    coord particle_velocity = {0., 0., 0.};  
  #endif  
  
  foreach()
    foreach_dimension()
      q.x[] = particles[]*RHO_S*particle_velocity.x + (1. - particles[])*q.x[];
  boundary ((scalar *){q});
  
  printf("\nEnd immersed_boundary (i++) pid:%d", pid());
  
}


event acceleration (i++) {
  
  printf("\nStart acceleration (i++) pid:%d", pid());
  
  face vector av = a;
  foreach_face(y)
    av.y[] -= GRAVITY;  
  
  printf("\nEnd acceleration (i++) pid:%d", pid());
  
}


event adapt (i++) {
  
  printf("\nStart adapt (i++) pid:%d", pid());
  
  vector u[];
  foreach()
    foreach_dimension()
      u.x[] = q.x[]/rho[];
  boundary((scalar *){u});

  #if dimension == 2
    adapt_wavelet({f, particles, u}, (double[]){1.e-3, 1.e-3, 1.e-2, 1.e-2}, LEVEL_MAX);
  #elif dimension == 3
    adapt_wavelet({f, particles, u}, (double[]){1.e-3, 1.e-3, 1.e-2, 1.e-2, 1.e-2}, LEVEL_MAX);
  #endif
    
  printf("\nEnd adapt (i++) pid:%d", pid());
    
}



double wet_area(scalar f, scalar particles) {
  
  printf("\nStart wet_area(scalar f, scalar particles) pid:%d", pid());

  // General orthogonal coordinates
  // http://basilisk.fr/src/README
  
  double wet = 0.;
  
  foreach(reduction(+:wet)) {    
    // By definition: solid=1, fluid=0
    if (particles[] >= 0.5) {      
      foreach_dimension() {        
        if (particles[1] < 0.5 && f[1] > 1e-6 )
          wet += fm.x[1]*Delta*f[1];
        if (particles[-1] < 0.5 && f[-1] > 1e-6 )
          wet += fm.x[]*Delta*f[-1];         
      }           
    }       
  }  
  
  return wet;  
  
  printf("\nEnd wet_area(scalar f, scalar particles) pid:%d", pid());
  
}


event compute_outputs (i=0; i++) {
  
  printf("\nStart compute_outputs (i=0; i++) pid:%d", pid());
  
  double wa = wet_area (f, particles);
  double la = interface_area (f);
  
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
  
  printf("\nEnd compute_outputs (i=0; i++) pid:%d", pid());
  
}


event console_log (i+=10) {
  
  printf("\nStart console_log (i+=10) pid:%d", pid());
  
  if (i%100 == 0)
    fprintf (ferr, "i t t* dt\n");
  fprintf (ferr, "%d %e %e %g\n", i, t, t*Vd/Dd, dt);
  
  printf("\nEnd console_log (i+=10) pid:%d", pid());
  
}


event simulation_log (i+=100; i<1000000) {
  
  printf("\nStart simulation_log (i+=100; i<1000000) pid:%d", pid());
  
  static FILE *simulationLog = fopen ("./logs/simulationLog.csv", "w");
  //printf("simulationLog = fopen()\n");
  
  if (i == 0)
    fprintf (simulationLog, "run, t, dt, mgp.i, mgu.i, grid->tn, perf.t, perf.speed\n");  
  fprintf (simulationLog, "%d, %g, %g, %d, %d, %ld, %g, %g\n", current_run, t, dt, mgp.i, mgu.i, grid->tn, perf.t, perf.speed);  
  fflush (simulationLog);  
  
  printf("\nEnd simulation_log (i+=100; i<1000000) pid:%d", pid());
  
}

event write_dump (i=0; i+=200) {
  
  printf("\nStart write_dump (i=0; i+=200) pid:%d", pid());
  
  char name[80];
  sprintf (name, "./dump/dump-%d-t-%.6g.ppm", current_run, t*Vd/Dd);
  dump (file = name); 
  
  printf("\nEnd write_dump (i=0; i+=200) pid:%d", pid());
  
}


event fields (i=200; i+=200) {
  
  printf("\nStart fields (i=200; i+=200) pid:%d", pid());
  
  char ppm_name_vof[80];
  sprintf(ppm_name_vof, "./images/vof-run-%d-t-%.6g-pid-%d.ppm", current_run, t*Vd/Dd, pid());
          
  char ppm_name_level[80]; 
  sprintf(ppm_name_level, "./images/level-run-%d-t-%.6g-pid-%d.ppm", current_run, t*Vd/Dd, pid());
    
  char ppm_name_vort[80]; 
  sprintf(ppm_name_vort, "./images/vort-run-%d-t-%.6g-pid-%d.ppm", current_run, t*Vd/Dd, pid());
  
  scalar ff[], ll[], m[], omega[];
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
  
  #if dimension == 2
    vorticity (u, omega);
  #elif dimension == 3
    lambda2 (u, omega);
  #endif 
    
  output_ppm(ff, file=ppm_name_vof, mask=m, n=1<<LEVEL_MAX);
  output_ppm(ll, file=ppm_name_level, max=LEVEL_MAX, n=1<<LEVEL_MAX);
  output_ppm(omega, file=ppm_name_vort, mask=m, n=1<<LEVEL_MAX);
    
  #if _OPENMP || _MPI
    char ppm_name_pid[80];
    sprintf(ppm_name_pid, "./images/pid-run-%d-t-%.6g-pid-%d.ppm", current_run, t*Vd/Dd, pid());
    scalar pid[];
    foreach()
      pid[] = tid();
    double tmax = npe() - 1;
    output_ppm (pid, file=ppm_name_pid, max=tmax, n=1<<LEVEL_MAX);
  #endif
    
  printf("\nEnd fields (i=200; i+=200) pid:%d", pid());
   
}



// Periodic boundary conditions at main () 
// Check compatibility at http://basilisk.fr/src/COMPATIBILITY

// #ifdef 0
// q.n[bottom] = q.n[] < 0 ? neumann(0) : 0;
// p[bottom] = dirichlet(PRESSURE);

// q.n[top] = neumann(0);
// p[top] = dirichlet(PRESSURE);

// q.n[right] = neumann(0);
// p[right] = dirichlet(PRESSURE);

// q.n[left] = neumann(0);
// p[left] = dirichlet(PRESSURE);
// #endif
