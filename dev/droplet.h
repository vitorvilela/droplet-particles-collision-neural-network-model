#ifndef __DROPLET_H__
#define __DROPLET_H__

#include <stdlib.h>
#include <string.h>


#define MAX_PARTICLES 4


void write_dataset_header(FILE *dataset) {   
  
  #if dimension == 2
    fprintf (dataset, "run,Re,We,Np,Csg,Cls,Als,D1,D2,D3,D4,X1,Y1,X2,Y2,X3,Y3,X4,Y4,MEAN(Wa),STD(Wa),MEAN(La),STD(La)\n");
  #elif dimension == 3
    fprintf (dataset, "run,Re,We,Np,Csg,Cls,Als,D1,D2,D3,D4,X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3,X4,Y4,Z4,MEAN(Wa),STD(Wa),MEAN(La),STD(La)\n");
  #endif  
  
  fflush (dataset);    
  
}


#if dimension == 2
void write_dataset_input(FILE *dataset, int nr, int Re, int We, int number_of_particles, double Dp[], double Csg, double Cls, double Als, double Xp[], double Yp[]) {
#elif dimension == 3
void write_dataset_input(FILE *dataset, int nr, int Re, int We, int number_of_particles, double Dp[], double Csg, double Cls, double Als, double Xp[], double Yp[], double Zp[]) {  
#endif
  
  char diameters[2000] = "";
  char diameter[200] = "";
  for (int np=0; np<MAX_PARTICLES; np++) {        
    if (np<MAX_PARTICLES-1)
      sprintf (diameter, "%.9g, ", Dp[np]);
    else
      sprintf (diameter, "%.9g", Dp[np]);   
    strcat(diameters, diameter);                
  }    
                 
  char positions[8000] = "";
  char position[2000] = "";
  for (int np=0; np<MAX_PARTICLES; np++) {
    #if dimension == 2
      sprintf (position, "%.9g, %.9g, ", Xp[np], Yp[np]);   
    #elif dimension == 3  
      sprintf (position, "%.9g, %.9g, %.9g, ", Xp[np], Yp[np], Zp[np]);   
    #endif  
    strcat(positions, position);      
  }

  fprintf (dataset, "%d, %d, %d, %d, %g, %g, %g, %s, %s", nr, Re, We, number_of_particles, Csg, Cls, Als, diameters, positions);  
  fflush (dataset);
  
}



void write_dataset_output(FILE *dataset, double mean_x1, double std_x1, double mean_x2, double std_x2) {
  
  // The variance of a sum of two random variables is given by
  // Var(aX - bY) = a²Var(X) + b²Var(Y) - 2abCov(X,Y)
  // https://en.wikipedia.org/wiki/Variance
  // It can be used further when analysing X1 and X2 outputs
  
  fprintf (dataset, "%.9g, %.9g, %.9g, %.9g\n", mean_x1, std_x1, mean_x2, std_x2);
  fflush (dataset);
  
}
  


#if dimension == 2  
int check_overlap(int np, double Dp[], double Xp[], double Yp[]) {
#elif dimension == 3  
int check_overlap(int np, double Dp[], double Xp[], double Yp[], double Zp[]) {
#endif
  
  printf("\nStart check_overlap() pid:%d", pid());
  
  if (np == 0) 
    return 0;
  
  for (int p=0; p<np; p++) {
    
    printf("\ndouble Xp[p]:%f pid:%d", Xp[p], pid());
    printf("\ndouble Yp[p]:%f pid:%d", Yp[p], pid());
    
    double distance = 0.;
    
    double x = Xp[p] - Xp[np];
    double y = Yp[p] - Yp[np];   
        
    #if dimension == 2
      distance = sqrt(x*x + y*y);
    #elif dimension == 3
      printf("\ndouble Zp[p]:%f pid:%d", Zp[p], pid());
      double z = Zp[p] - Zp[np];
      distance = sqrt(x*x + y*y + z*z);
    #endif  
      
    if ( distance > (0.5*Dp[p] + 0.5*Dp[np]) )
      continue;
    else {
      printf("\nEnd check_overlap() pid:%d", pid());
      return 1;     
    }
    
  }

  printf("\nEnd check_overlap() pid:%d", pid());
  
  return 0;  

}  

  
#endif
