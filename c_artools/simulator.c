#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

#define NUM_LAYERS 5

/* The main mathematical functionality of the simulator.py module written in C. 
Just for shits and giggles. And learning C.

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
double _Complex * calc_RT_amp(double frequency, double dielectric[NUM_LAYERS], 
			      double loss[NUM_LAYERS], double thickness[NUM_LAYERS]);

// main, which is really the debugging portion

int main() {
  printf("\n#####################\n## BEGIN DEBUGGING ##\n#####################\n\n");

/*   double dielectric1; */
/*   double dielectric2; */
/*   double index1; */
/*   double index2; */
/*   dielectric1 = 4; */
/*   dielectric2 = 7; */
/*   printf("Test 'get_index'\n"); */
/*   index1 = get_index(dielectric1); */
/*   index2 = get_index(dielectric2); */
/*   printf("'get_index' result is %lf\n\n", index1); */

/*   double freq; */
/*   double loss; */
/*   double _Complex k; */
/*   freq = 150E9; */
/*   loss = 0.00025; */
/*   printf("Test 'find_k'\n"); */
/*   k = find_k(index1, freq, loss); */
/*   printf("'find_k' result is %lf %lfi\n\n", creal(k), cimag(k)); */
  
/*   double d; */
/*   double _Complex delta; */
/*   d = 15.0*2.54E-5; */
/*   printf("Test 'find_k_offset'\n"); */
/*   delta = find_k_offset(k, d); */
/*   printf("'find_k_offset' result is %lf %lfi\n\n", creal(delta), cimag(delta)); */
  
/*   double r; */
/*   printf("Test 'r_at_interface'\n"); */
/*   r = r_at_interface(index1, index2); */
/*   printf("'r_at_interface' result is %lf\n\n", r); */

/*   double t; */
/*   printf("Test 't_at_interface'\n"); */
/*   t = t_at_interface(index1, index2); */
/*   printf("'t_at_interface' result is %lf\n\n", t); */

/*   double ** matrix; */
/*   printf("Test 'make_2x2'\n"); */
/*   matrix = make_2x2(); */
/*   printf("'make_2x2' result is:\n %lf %lf\n %lf %lf\n\n", */
/* 	 matrix[0][0], matrix[0][1], */
/* 	 matrix[1][0], matrix[1][1]); */
/*   free(matrix); */

/*   double _Complex ** c_matrix; */
/*   printf("Test 'make_complex_2x2'\n"); */
/*   c_matrix = make_complex_2x2(); */
/*   printf("'make_complex_2x2' result is:\n%lf+%lfi  %lf+%lfi\n%lf+%lfi  %lf+%lfi\n\n", */
/* 	 creal(c_matrix[0][0]), cimag(c_matrix[0][0]), */
/* 	 creal(c_matrix[0][1]), cimag(c_matrix[0][1]), */
/* 	 creal(c_matrix[1][0]), cimag(c_matrix[1][0]), */
/* 	 creal(c_matrix[1][1]), cimag(c_matrix[1][1])); */

/*   printf("Rewrite 'make_complex_2x2' values\n"); */
/*   c_matrix[0][0] = 42.0 - 2.0*I; */
/*   c_matrix[0][1] = 22.43 + 9.0*I; */
/*   c_matrix[1][0] = 1.0 + 33.0*I; */
/*   c_matrix[1][1] = cexp(5.0+7777.7*I); */
/*   printf("Rewrite 'make_complex_2x2' result is:\n%lf+%lfi  %lf+%lfi\n%lf+%lfi  %lf+%lfi\n\n", */
/* 	 creal(c_matrix[0][0]), cimag(c_matrix[0][0]), */
/* 	 creal(c_matrix[0][1]), cimag(c_matrix[0][1]), */
/* 	 creal(c_matrix[1][0]), cimag(c_matrix[1][0]), */
/* 	 creal(c_matrix[1][1]), cimag(c_matrix[1][1])); */
/*   free(c_matrix); */

/*   double _Complex r_amp; */
/*   double R; */
/*   r_amp = 2.5 + 4.0*I; */
/*   printf("Test 'get_R'\n"); */
/*   R = get_R(r_amp); */
/*   printf("The result of 'get_R' is %lf\n\n", R); */

/*   double _Complex t_amp; */
/*   double T; */
/*   double n_i; */
/*   double n_f; */
/*   n_i = 1.0; */
/*   n_f = 9.7; */
/*   t_amp = 258. + 77.0*I; */
/*   printf("Test 'get_T'\n"); */
/*   T = get_T(t_amp, n_i, n_f); */
/*   printf("The result of 'get_T' is %lf\n\n", T); */

/*   printf("Test adding stuff to an array\n"); */
/*   int array[100]; */
/*   int count; */
/*   array[0] = 42; */
/*   count = 0; */
/*   printf("The value in array[0] is %d\n\n", array[0]); */
  
/*   double _Complex *** c_array_3d; */
/*   int n; */
/*   n = 3; */
/*   printf("Test 'make_complex_nd' with dimension 3\n"); */
/*   c_array_3d = make_complex_nd(n); */
/*   printf("Slice 1:\n%lf+%lfi %lf+%lfi\n%lf+%lfi %lf+%lfi\n", */
/* 	 creal(c_array_3d[0][0][0]), cimag(c_array_3d[0][0][0]), */
/* 	 creal(c_array_3d[0][0][1]), cimag(c_array_3d[0][0][1]), */
/* 	 creal(c_array_3d[0][1][0]), cimag(c_array_3d[0][1][0]), */
/* 	 creal(c_array_3d[0][1][1]), cimag(c_array_3d[0][1][1])); */
/*   printf("Slice 2:\n%lf+%lfi %lf+%lfi\n%lf+%lfi %lf+%lfi\n", */
/* 	 creal(c_array_3d[1][0][0]), cimag(c_array_3d[1][0][0]), */
/* 	 creal(c_array_3d[1][0][1]), cimag(c_array_3d[1][0][1]), */
/* 	 creal(c_array_3d[1][1][0]), cimag(c_array_3d[1][1][0]), */
/* 	 creal(c_array_3d[1][1][1]), cimag(c_array_3d[1][1][1])); */
/*   printf("Slice 3:\n%lf+%lfi %lf+%lfi\n%lf+%lfi %lf+%lfi\n", */
/* 	 creal(c_array_3d[2][0][0]), cimag(c_array_3d[2][0][0]), */
/* 	 creal(c_array_3d[2][0][1]), cimag(c_array_3d[2][0][1]), */
/* 	 creal(c_array_3d[2][1][0]), cimag(c_array_3d[2][1][0]), */
/* 	 creal(c_array_3d[2][1][1]), cimag(c_array_3d[2][1][1])); */

/*   // make a mock simulation using the inputs: */
/*   // thickness, dielectric constant, and loss tangent */

  printf("\n>>>> Testing grabbing from arrays and calculating things, then putting values back into new arrays. <<<<\n\n");
  double freq;
  double dielectric[NUM_LAYERS] = {1.0, 2.4, 3.5, 6.15, 9.7};
  double thickness[NUM_LAYERS] = {1000.0, 15.0*2.54E-5, 5.0*2.54E-5, 5.0*2.54E-5, 1000.0};
  double loss[NUM_LAYERS] = {0.0, 2.5E-4, 1.7E-3, 1.526E-3, 7.4E-4};

  freq = 150E9;
/*   double _Complex ks[NUM_LAYERS]; */
/*   double deltas[NUM_LAYERS]; */
/*   double index[NUM_LAYERS]; */
/*   // int count; */
/*   int i; */

/*   printf("Test converting the array of dielectric constants to refractive indices.\n"); */
/*   for (i=0; i<NUM_LAYERS; i++) { */
/*     index[i] = get_index(dielectrics[i]); */
/*   } */
/*   printf("The conversion from dielectric to index is:\n"); */
/*   for (i=0; i<NUM_LAYERS; i++) { */
/*     printf("%lf ---> %lf\n", dielectrics[i], index[i]); */
/*   } */

/*   printf("\nTest calculating the wavenumbers.\n"); */
/*   for (i=0; i<NUM_LAYERS; i++) { */
/*     ks[i] = find_k(index[i], freq, losses[i]); */
/*   } */
/*   printf("The dimensioned wavenumber (with units m^(-1)) for each layer is:\n"); */
/*   for (i=0; i<NUM_LAYERS; i++) { */
/*     printf("%lf+%lfi\n", creal(ks[i]), cimag(ks[i])); */
/*   } */
/*   for (i=0; i<NUM_LAYERS; i++) { */
/*     deltas[i] = ks[i]*thicknesses[i]; */
/*   } */
/*   printf("\nAnd the dimensionless wavenumber for each layer is:\n"); */
/*   for (i=0; i<NUM_LAYERS; i++) { */
/*     printf("%lf+%lf\n", creal(deltas[i]), cimag(deltas[i])); */
/*   } */

  double _Complex * test_val;
  test_val = calc_RT_amp(freq, dielectric, loss, thickness);

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
  return fabs(pow(t_amp, 2))*(n_f/n_i);
  //return pow(cabs(t_amp), 2)*(n_f/n_i);
}

double _Complex * calc_RT_amp(double frequency, double dielectric[NUM_LAYERS], double loss[NUM_LAYERS], double thickness[NUM_LAYERS]) {
  double index[NUM_LAYERS];
  double _Complex delta[NUM_LAYERS];
  double _Complex r_amp[NUM_LAYERS][NUM_LAYERS] = {0};
  double _Complex t_amp[NUM_LAYERS][NUM_LAYERS] = {0};
  double _Complex *** M;
  double _Complex ** M_prime;
  double _Complex ** M2_prime;
  double _Complex *** m_r_amp;
  double _Complex *** m_t_amp;
  double _Complex *** m_rt_prod;
  double _Complex * RT_amp;
  double _Complex sum;
  //  int testA[2][2] = {1,2,3,4};
  //  int testB[2][2] = {5,6,7,8};
  //  int testProd[2][2] = {0};
  //  int i,j,k,l;

  printf("\n#### Inside 'calc_RT_amp'. ####\n");

  RT_amp = (double _Complex *)malloc(sizeof(double _Complex)*2);

  for (int i=0; i<NUM_LAYERS; i++) {
    index[i] = get_index(dielectric[i]);
    delta[i] = find_k(frequency, index[i], loss[i], thickness[i]);
  }

  // print statements for debugging
  printf("\nHere are the refractive indices calculated from the dielectric constants passed to the function.\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    printf("%lf\n", index[i]);
  }
  printf("\nHere are the deltas calculated from the input frequency, indices, losses, and thicknesses.\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    printf("%lf+%lfi\n", creal(delta[i]), cimag(delta[i]));
  }
  printf("\nHere are the original values of 'r_amp' and 't_amp'.\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<NUM_LAYERS; j++) {
      printf("r_amp %d%d ---> %lf %lfi\n", i, j, creal(r_amp[i][j]), cimag(r_amp[i][j]));
	}
  }
  printf("\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<NUM_LAYERS; j++) {
      printf("t_amp %d%d ---> %lf %lfi\n", i, j, creal(t_amp[i][j]), cimag(t_amp[i][j]));
    }
  }

  for (int i=0; i<NUM_LAYERS-1; i++) {
    r_amp[i][i+1] = r_at_interface(index[i], index[i+1]);
    t_amp[i][i+1] = t_at_interface(index[i], index[i+1]);
  }
  printf("\nHere are the modified values of 'r_amp' and 't_amp'.\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<NUM_LAYERS; j++) {
      printf("mod r_amp %d%d ---> %lf %lfi\n", i, j, creal(r_amp[i][j]), cimag(r_amp[i][j]));
    }
  }
  printf("\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<NUM_LAYERS; j++) {
      printf("mod t_amp %d%d ---> %lf %lfi\n", i, j, creal(t_amp[i][j]), cimag(t_amp[i][j]));
    }
  }

  M = make_complex_nd(NUM_LAYERS);
  printf("\nThe 'M' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	printf("M%d%d%d ---> %lf %lf\n", i, j, k, creal(M[i][j][k]), cimag(M[i][j][k]));
      }
    }
  }

  m_r_amp = make_complex_nd(NUM_LAYERS);
  m_t_amp = make_complex_nd(NUM_LAYERS);
  printf("\nThe temporary 'm_r_amp' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	printf("m_r_amp%d%d%d ---> %lf %lf\n", i, j, k, creal(m_r_amp[i][j][k]), cimag(m_r_amp[i][j][k]));
      }
    }
  }  
  printf("\nThe temporary 'm_t_amp' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	printf("m_t_amp%d%d%d ---> %lf %lf\n", i, j, k, creal(m_t_amp[i][j][k]), cimag(m_t_amp[i][j][k]));
      }
    }
  }

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

  printf("\nThe modified temporary 'm_r_amp' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        printf("mod m_r_amp%d%d%d ---> %lf %lf\n", i, j, k, creal(m_r_amp[i][j][k]), cimag(m_r_amp[i][j][k]));
      }
    }
  }

  printf("\nThe modified temporary 'm_t_amp' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        printf("mod m_t_amp%d%d%d ---> %lf %lf\n", i, j, k, creal(m_t_amp[i][j][k]), cimag(m_t_amp[i][j][k]));
      }
    }
  }

  m_rt_prod = make_complex_nd(NUM_LAYERS);
  printf("\nThe temporary 'm_rt_prod' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        printf("m_rt_prod%d%d%d ---> %lf %lf\n", i, j, k, creal(m_rt_prod[i][j][k]), cimag(m_rt_prod[i][j][k]));
      }
    }
  }

  printf("\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	sum = 0;
	for (int l=0; l<2; l++) {
	  //	  printf("multiplying m_t_amp%d%d%d by m_r_amp%d%d%d\n", i, j, l, i, l, k);
	  sum = sum + (m_t_amp[i][j][l]*m_r_amp[i][l][k]);
	}
	m_rt_prod[i][j][k] = sum;
	M[i][j][k] = sum;
      }
    }
  }
  printf("\nThe modified temporary 'm_rt_prod' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        printf("mod m_rt_prod%d%d%d ---> %lf %lf\n", i, j, k, creal(m_rt_prod[i][j][k]), cimag(m_rt_prod[i][j][k]));
      }
    }
  }

  printf("\nThe first modified 'M' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        printf("1st mod M%d%d%d ---> %lf %lf\n", i, j, k, creal(M[i][j][k]), cimag(M[i][j][k]));
      }
    }
  }

  for (int i=1; i<NUM_LAYERS-1; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	M[i][j][k] = M[i][j][k]*(1/t_amp[i][i+1]);
      }
    }
  }

  printf("\nThe second modified 'M' matrix is:\n");
  for (int i=0; i<NUM_LAYERS; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
        printf("2nd mod M%d%d%d ---> %lf %lf\n", i, j, k, creal(M[i][j][k]), cimag(M[i][j][k]));
      }
    }
  }

  M_prime = make_complex_2x2();
  M2_prime = make_complex_2x2();
  printf("\nThe 'M_prime' matrix is:\n");
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
        printf("M_prime%d%d ---> %lf %lf\n", i, j, creal(M_prime[i][j]), cimag(M_prime[i][j]));
    }
  }
  printf("\nThe 'M2_prime' matrix is:\n");
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      printf("M2_prime%d%d ---> %lf %lf\n", i, j, creal(M2_prime[i][j]), cimag(M2_prime[i][j]));
    }
  }

  M_prime[0][0] = 1.0;
  M_prime[1][1] = 1.0;
  printf("\nThe first modified 'M_prime' matrix is:\n");
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      printf("1st mod M_prime%d%d ---> %lf %lf\n", i, j, creal(M_prime[i][j]), cimag(M_prime[i][j]));
    }
  }

  for (int i=1; i<NUM_LAYERS-1; i++) {
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	sum = 0;
	for (int l=0; l<2; l++) {
	  printf("\n\n##################\nM_prime%d%d and M%d%d%d\n##################", j, l, i, l, k);
	  printf("\n'sum A' is ---> %lf %lfi", creal(sum), cimag(sum));
	  printf("\nMultiplying %lf %lfi by %lf %lfi", 
		 creal(M_prime[j][l]), cimag(M_prime[j][l]), creal(M[i][l][k]), cimag(M[i][l][k]));
	  sum = sum + M_prime[j][l]*M[i][l][k];
	  printf("\n'sum B' is ---> %lf %lfi", creal(sum), cimag(sum));
	}
	M2_prime[j][k] = sum;
	printf("\n\n>>>>>'M2_prime%d%d #%d' is ---> %lf %lfi <<<<<\n", 
	       j, k, i, creal(M2_prime[j][k]), cimag(M2_prime[j][k]));
      }
    }
    for (int j=0; j<2; j++) {
      for (int k=0; k<2; k++) {
	M_prime[j][k] = M2_prime[j][k];
	M2_prime[j][k] = 0.;
      }
    }
  }

  printf("\nThe second modified 'M_prime' matrix is:\n");
  for (int i=0; i<2; i++) {
    for (int j=0; j<2; j++) {
      printf("2nd mod M_prime%d%d ---> %lf %lf\n", 
	     i, j, creal(M_prime[i][j]), cimag(M_prime[i][j]));
    }
  }

/*   printf("\nSimple 2x2 matrix multiplication example\n"); */
/*   printf("The inputs for matrix A are:\n"); */
/*   for (int i=0; i<2; i++) { */
/*     for (int j=0; j<2; j++) { */
/*       printf("A%d%d ---> %d\n", i, j, testA[i][j]); */
/*     } */
/*   } */
/*   printf("\nThe inputs for matrix B are:\n"); */
/*   for (int i=0; i<2; i++) { */
/*     for (int j=0; j<2; j++) { */
/*       printf("A%d%d ---> %d\n", i, j, testB[i][j]); */
/*     } */
/*   } */

/*   for (int i=0; i<2; i++) { */
/*     for (int j=0; j<2; j++) { */
/*       sum = 0; */
/*       for (int k=0; k<2; k++) { */
/* 	sum = sum + testA[i][k]*testB[k][j]; */
/*       } */
/*       testProd[i][j] = sum; */
/*     } */
/*   } */

/*   printf("\nThe product of A and B is:\n"); */
/*   for (int i=0; i<2; i++) { */
/*     for (int j=0; j<2; j++) { */
/*       printf("AB%d%d ---> %d\n", i, j, testProd[i][j]); */
/*     } */
/*   } */

  RT_amp[0] = -1.0-1.0*I;
  printf("\n#### Leaving 'calc_RT_amp' ####\n");
  return RT_amp;
}
