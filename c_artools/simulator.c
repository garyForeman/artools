#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

#define NUM_LAYERS 5
#define VERBOSE 0

/* The main mathematical functionality of the simulator.py module written in C. 
Just for shits and giggles. And for learning C.

Debugging will happen in this file until I can figure out how to get files to 
talk to one another.*/


// function prototyping

double get_index(double dielectric);
double _Complex find_k(double frequency, double index, double loss, double thickness);
//double _Complex find_k_offset(double _Complex k, double d);
double r_at_interface(double n1, double n2);
double t_at_interface(double n1, double n2);
double ** make_2x2(void);
double _Complex ** make_complex_2x2(void);
double _Complex *** make_complex_nd(int n);
double get_R(double _Complex r_amp);
double get_T(double _Complex t_amp, double n_i, double n_f);
double _Complex * calc_rt_amp(double frequency, double dielectric[NUM_LAYERS], 
			      double loss[NUM_LAYERS], double thickness[NUM_LAYERS]);

// main, which is really the debugging portion

int main() {
  printf("\n#####################\n## BEGIN DEBUGGING ##\n#####################\n\n");
  double freq;
  double dielectric[NUM_LAYERS] = {1.0, 2.4, 3.5, 6.15, 9.7};
  double thickness[NUM_LAYERS] = {1000.0, 15.0*2.54E-5, 5.0*2.54E-5, 5.0*2.54E-5, 1000.0};
  double loss[NUM_LAYERS] = {0.0, 2.5E-4, 1.7E-3, 1.526E-3, 7.4E-4};

  freq = 150E9;
  double index[NUM_LAYERS];

  for (int i=0; i<NUM_LAYERS; i++) {
    index[i] = get_index(dielectric[i]);
  }

  double _Complex * rt_amp;
  rt_amp = calc_rt_amp(freq, dielectric, loss, thickness);

  //  printf("\nt amplitude is ---> %lf %lfi", creal(rt_amp[0]), cimag(rt_amp[0]));
  //  printf("\nr amplitude is ---> %lf %lfi", creal(rt_amp[1]), cimag(rt_amp[1]));

  double T;
  double R;

  T = get_T(rt_amp[0], index[0], index[NUM_LAYERS-1]);
  R = get_R(rt_amp[1]);

  //  printf("\n\nPercent transmission is ---> %lf", T);
  //  printf("\nPercent reflection is ---> %lf", R);

  printf("\n\n#####################\n### END DEBUGGING ###\n#####################\n\n");
}

double get_index(double dielectric) {
  return sqrt(dielectric);
}

double _Complex find_k(double frequency, double index, double loss, double thickness) {
  return (2*M_PI*index*frequency*thickness*(1-0.5*loss*I))/3E8;
}

double _Complex find_k_offset(double _Complex k, double d) {
  return k*d;
}

double r_at_interface(double n1, double n2) {
  return (n1-n2)/(n1+n2);
}

double t_at_interface(double n1, double n2) {
  return (2*n1)/(n1+n2);
}

double ** make_2x2(void) {
  double ** matrix;
  int val;
  val = 0;
  matrix = (double **)malloc(sizeof(double *)*2);
  for (int i=0; i<2; i++) {
    matrix[i] = (double *)malloc(sizeof(double)*2);
    for (int j=0; j<2; j++) {
      matrix[i][j] = val;
    }
  }
  return matrix;
}

double _Complex ** make_complex_2x2(void) {
  double _Complex ** matrix;
  int val;
  val = 0;
  matrix = (double _Complex **)malloc(sizeof(double _Complex *)*2);
  for (int i=0; i<2; i++) {
    matrix[i] = (double _Complex *)malloc(sizeof(double _Complex)*2);
    for (int j=0; j<2; j++) {
      matrix[i][j] = val;
    }
  }
  return matrix;
}

double _Complex *** make_complex_nd(int n) {
  double _Complex *** nd_matrix;
  int val;
  val = 0;
  nd_matrix = (double _Complex ***)malloc(sizeof(double _Complex **)*n);
  for (int i=0; i<n; i++) {
    nd_matrix[i] = (double _Complex **)malloc(sizeof(double _Complex *)*2);
    for (int j=0; j<2; j++) {
      nd_matrix[i][j] = (double _Complex *)malloc(sizeof(double _Complex)*2);
      for (int k=0; k<2; k++) {
	nd_matrix[i][j][k] = val;
      }
    }
  }
  return nd_matrix;
}

double get_R(double _Complex r_amp) {
  return pow(cabs(r_amp), 2);
}

double get_T(double _Complex t_amp, double n_i, double n_f) {
  // should this actually be the commented out expression????
  // in Python I wrote it like this:
  // return np.abs(net_t_amp**2)*(n_f/n_i)
  // but maybe that isn't correct? second guessing myself here...
  // return fabs(pow(t_amp, 2))*(n_f/n_i); // <--- this version is almost certainly wrong
  return pow(cabs(t_amp), 2)*(n_f/n_i);
}

double _Complex * calc_rt_amp(double frequency, double dielectric[NUM_LAYERS], double loss[NUM_LAYERS], double thickness[NUM_LAYERS]) {
  double index[NUM_LAYERS];
  double _Complex delta[NUM_LAYERS];
  double _Complex r_amp[NUM_LAYERS][NUM_LAYERS] = {0};
  double _Complex t_amp[NUM_LAYERS][NUM_LAYERS] = {0};
  double _Complex *** M;
  double _Complex ** M_prime;
  double _Complex ** M2_prime;
  double _Complex ** M_final;
  double _Complex *** m_r_amp;
  double _Complex *** m_t_amp;
  double _Complex *** m_rt_prod;
  double _Complex * rt_amp;
  double _Complex sum;

  if (VERBOSE) {
    printf("\n#### Entering 'calc_rt_amp'. ####\n");
  }

  rt_amp = (double _Complex *)malloc(sizeof(double _Complex)*2);

  if (VERBOSE) {
    printf("\nCalculating the refractive indices and wavenumbers for each layer.\n");
  }
  for (int i=0; i<NUM_LAYERS; i++) {
    //    printf("Calling 'get_index' on layer%d", i);
    index[i] = get_index(dielectric[i]);
    //    printf("\nCalling 'find_k' on layer%d\n", i);
    delta[i] = find_k(frequency, index[i], loss[i], thickness[i]);
  }

  if (VERBOSE) {
    printf("\nHere are the refractive indices calculated from the dielectric constants passed to the function.\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      printf("%lf\n", index[i]);
    }
    printf("\nHere are the wavenumbers calculated from the input frequency, indices, losses, and thicknesses.\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      printf("%lf+%lfi\n", creal(delta[i]), cimag(delta[i]));
    }
  }

  if (VERBOSE) {
    printf("\nCalculating the reflection and transmission coefficients at each interface.\n");
  }
  for (int i=0; i<NUM_LAYERS-1; i++) {
    //    printf("Calling 'r_at_interface' for layer%d/layer%d interface", i, i+1);
    r_amp[i][i+1] = r_at_interface(index[i], index[i+1]);
    //    printf("\nCalling 't_at_interface' for layer%d/layer%d interface\n", i, i+1);
    t_amp[i][i+1] = t_at_interface(index[i], index[i+1]);
    }

  if (VERBOSE) {
    printf("\nThe reflection coefficient matrix is:\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      for (int j=0; j<NUM_LAYERS; j++) {
	printf("Reflection matrix element %d%d ---> %lf %lfi\n", i, j, creal(r_amp[i][j]), cimag(r_amp[i][j]));
      }
    }
    printf("\nThe transmission coefficient matrix is:\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      for (int j=0; j<NUM_LAYERS; j++) {
	printf("Reflection matrix element %d%d ---> %lf %lfi\n", i, j, creal(t_amp[i][j]), cimag(t_amp[i][j]));
      }
    }
  }

  if (VERBOSE) {
    printf("\nCreating %d 2x2 matrices to store transmission and reflection values\n", NUM_LAYERS);
  }

  M = make_complex_nd(NUM_LAYERS);

  if (VERBOSE) {
    printf("\nCreating %d temporary 2x2 matrices for both reflection and transmission at each layer\n", NUM_LAYERS);
  }

  m_r_amp = make_complex_nd(NUM_LAYERS);
  m_t_amp = make_complex_nd(NUM_LAYERS);

  for (int i=1; i<NUM_LAYERS-1; i++) {
    m_t_amp[i][0][0] = cexp(-1*delta[i]*I);
    m_t_amp[i][0][1] = 0.0;
    m_t_amp[i][1][0] = 0.0;
    m_t_amp[i][1][1] = cexp(delta[i]*I);
    m_r_amp[i][0][0] = 1.0;
    m_r_amp[i][0][1] = r_amp[i][i+1];
    m_r_amp[i][1][0] = r_amp[i][i+1];
    m_r_amp[i][1][1] = 1.0;
  }

  if (VERBOSE) {
    printf("\nThe values of elements in the temporary reflection matrices are:\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      printf("Temporary reflection matrix %d\n", i);
      for (int j=0; j<2; j++) {
	for (int k=0; k<2; k++) {
	  printf("%d%d ---> %lf %lfi\n", j, k, creal(m_r_amp[i][j][k]), cimag(m_r_amp[i][j][k]));
	}
      }
    }

    printf("\nThe values of elements in the temporary transmission matrices are:\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      printf("Temporary transmission matrix %d\n", i);
      for (int j=0; j<2; j++) {
	for (int k=0; k<2; k++) {
	  printf("%d%d ---> %lf %lfi\n", j, k, creal(m_t_amp[i][j][k]), cimag(m_t_amp[i][j][k]));
	}
      }
    }
  }

  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	sum = 0;
	for (int l=0; l<2; l++) {
	  //	  printf("multiplying m_t_amp%d%d%d by m_r_amp%d%d%d\n", i, j, l, i, l, k);
	  sum = sum + (m_t_amp[i][j][l]*m_r_amp[i][l][k]);
	}
	//	m_rt_prod[i][j][k] = sum;
	M[i][j][k] = sum;
      }
    }
  }

  if (VERBOSE) {
    printf("\nThe M matrices, the products of the temporary transmission matrices by reflection matrices, are:\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      printf("M%d, the product of transmission%d and reflection%d\n", i, i, i);
      for (int j=0; j<2; j++) {
	for (int k=0; k<2; k++) {
	  printf("%d%d ---> %lf %lfi\n", j, k, creal(M[i][j][k]), cimag(M[i][j][k]));
	}
      }
    }
  }

  if (VERBOSE) {
    printf("\nScaling the M matrices by the the reciprocal of the corresponding interface transmission coefficient\n");
  }
  for (int i=1; i<NUM_LAYERS-1; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	M[i][j][k] = M[i][j][k]*(1/t_amp[i][i+1]);
      }
    }
  }

  if (VERBOSE) {
    printf("\nThe scaled M matrices are:\n");
    for (int i=0; i<NUM_LAYERS; i++) {
      printf("M%d (scaled)\n", i);
      for (int j=0; j<2; j++) {
	for (int k=0; k<2; k++) {
	  printf("%d%d ---> %lf %lfi\n", j, k, creal(M[i][j][k]), cimag(M[i][j][k]));
	}
      }
    }
  }

  if (VERBOSE) {
    printf("\nCreating the 'M_prime' matrix\n");
  }
  M_prime = make_complex_2x2();
  if (VERBOSE) {
    printf("\nCreating the temporary 'M2_prime' matrix\n");
  }
  M2_prime = make_complex_2x2();

  M_prime[0][0] = 1.0;
  M_prime[1][1] = 1.0;

  if (VERBOSE) {
    printf("\nThe elements of the 'M_prime' matrix are:\n");
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
	printf("%d%d ---> %lf %lf\n", i, j, creal(M_prime[i][j]), cimag(M_prime[i][j]));
      }
    }
  }

  if (VERBOSE) {
    printf("\nMultiplying the M_prime matrix by the M matrices\n");
  }
  for (int i=1; i<NUM_LAYERS-1; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	sum = 0;
	for (int l=0; l<2; l++) {
	  sum = sum + M_prime[j][l]*M[i][l][k];
	}
	M2_prime[j][k] = sum;
      }
    }
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	M_prime[j][k] = M2_prime[j][k];
	M2_prime[j][k] = 0.;
      }
    }
  }

  if (VERBOSE) {
    printf("\nThe result of the M_prime matrix by M matrix is:\n");
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
	printf("M_prime%d%d ---> %lf %lfi\n", 
	       i, j, creal(M_prime[i][j]), cimag(M_prime[i][j]));
      }
    }
  }

  if (VERBOSE) {
    printf("\nRewriting the values of the temporary M2_prime matrix so that \nelements 00 and 11 are the reciprocal of the transmission \ncoefficient at the first interface, and elements 01 and 10 are \nthe reflection coefficient at the first interface divided \nby the transmission coefficient at the first interface\n"); 
  }

  M2_prime[0][0] = 1./t_amp[0][1];
  M2_prime[0][1] = r_amp[0][1]/t_amp[0][1];
  M2_prime[1][0] = r_amp[0][1]/t_amp[0][1];
  M2_prime[1][1] = 1./t_amp[0][1];

  if (VERBOSE) {
    printf("\nThe elements of the rewritten M2_prime matrix are:\n");
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
	printf("%d%d ---> %lf %lfi\n",
	       i, j, creal(M2_prime[i][j]), cimag(M2_prime[i][j]));
      }
    }
  
    printf("\nCreating the final result matrix, M_final, the product of M2_prime and M_prime\n");
  }

  M_final = make_complex_2x2();

  if (VERBOSE) {
    printf("\nMultiplying the M2_prime matrix by the M_prime matrix\n");
  }

  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      sum = 0;
      for (int l=0; l<2; l++) {
	sum = sum + M2_prime[i][l]*M_prime[l][j];
      }
      M_final[i][j] = sum;
    }
  }

  if (VERBOSE) {
    printf("\nThe elements of the M_final matrix are:\n");
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
	printf("%d%d ---> %lf %lfi\n", i, j, creal(M_final[i][j]), cimag(M_final[i][j]));
      }
    }
   
    printf("\nCalculating the transmission amplitude, t = 1/M_final00\n");
  }

  rt_amp[0] = 1./M_final[0][0];
  if (VERBOSE) {
    printf("\nCalculation the reflection amplitdue, r = M_final01/M_final00\n");
  }
  rt_amp[1] = M_final[0][1]/M_final[0][0];

  if (VERBOSE) {
    printf("\nt ---> %lf %lfi", creal(rt_amp[0]), cimag(rt_amp[0]));
    printf("\nr ---> %lf %lfi", creal(rt_amp[1]), cimag(rt_amp[1]));

    printf("\n\n#### Leaving 'calc_RT_amp' ####\n");
  }
  return rt_amp;
}

