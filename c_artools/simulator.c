#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

/* The main mathematical functionality of the simulator.py module written in C. 
Just for shits and giggles. And learning C.

Debugging will happen in this file until I can figure out how to get files to 
talk to one another.*/


// function prototyping

double get_index(double dielectric);
double _Complex find_k(double index, double freq, double loss);
double _Complex find_k_offset(double _Complex k, double d);
double r_at_interface(double n1, double n2);
double t_at_interface(double n1, double n2);
double ** make_2x2(void);
double _Complex ** make_complex_2x2(void);
double _Complex *** make_complex_nd(int n);
double get_R(double _Complex r_amp);
double get_T(double _Complex t_amp, double n_i, double n_f);

// main, which is really the debugging portion

int main() {
  printf("\n#####################\n## BEGIN DEBUGGING ##\n#####################\n\n");

  double dielectric1;
  double dielectric2;
  double index1;
  double index2;
  dielectric1 = 4;
  dielectric2 = 7;
  printf("Test 'get_index'\n");
  index1 = get_index(dielectric1);
  index2 = get_index(dielectric2);
  printf("'get_index' result is %lf\n\n", index1);

  double freq;
  double loss;
  double _Complex k;
  freq = 150E9;
  loss = 0.00025;
  printf("Test 'find_k'\n");
  k = find_k(index1, freq, loss);
  printf("'find_k' result is %lf %lfi\n\n", creal(k), cimag(k));
  
  double d;
  double _Complex delta;
  d = 15.0*2.54E-5;
  printf("Test 'find_k_offset'\n");
  delta = find_k_offset(k, d);
  printf("'find_k_offset' result is %lf %lfi\n\n", creal(delta), cimag(delta));
  
  double r;
  printf("Test 'r_at_interface'\n");
  r = r_at_interface(index1, index2);
  printf("'r_at_interface' result is %lf\n\n", r);

  double t;
  printf("Test 't_at_interface'\n");
  t = t_at_interface(index1, index2);
  printf("'t_at_interface' result is %lf\n\n", t);

  double ** matrix;
  printf("Test 'make_2x2'\n");
  matrix = make_2x2();
  printf("'make_2x2' result is:\n %lf %lf\n %lf %lf\n\n",
	 matrix[0][0], matrix[0][1],
	 matrix[1][0], matrix[1][1]);
  free(matrix);

  double _Complex ** c_matrix;
  printf("Test 'make_complex_2x2'\n");
  c_matrix = make_complex_2x2();
  printf("'make_complex_2x2' result is:\n%lf+%lfi  %lf+%lfi\n%lf+%lfi  %lf+%lfi\n\n",
	 creal(c_matrix[0][0]), cimag(c_matrix[0][0]),
	 creal(c_matrix[0][1]), cimag(c_matrix[0][1]),
	 creal(c_matrix[1][0]), cimag(c_matrix[1][0]),
	 creal(c_matrix[1][1]), cimag(c_matrix[1][1]));

  printf("Rewrite 'make_complex_2x2' values\n");
  c_matrix[0][0] = 42.0 - 2.0*I;
  c_matrix[0][1] = 22.43 + 9.0*I;
  c_matrix[1][0] = 1.0 + 33.0*I;
  c_matrix[1][1] = cexp(5.0+7777.7*I);
  printf("Rewrite 'make_complex_2x2' result is:\n%lf+%lfi  %lf+%lfi\n%lf+%lfi  %lf+%lfi\n\n",
	 creal(c_matrix[0][0]), cimag(c_matrix[0][0]),
	 creal(c_matrix[0][1]), cimag(c_matrix[0][1]),
	 creal(c_matrix[1][0]), cimag(c_matrix[1][0]),
	 creal(c_matrix[1][1]), cimag(c_matrix[1][1]));
  free(c_matrix);

  double _Complex r_amp;
  double R;
  r_amp = 2.5 + 4.0*I;
  printf("Test 'get_R'\n");
  R = get_R(r_amp);
  printf("The result of 'get_R' is %lf\n\n", R);

  double _Complex t_amp;
  double T;
  double n_i;
  double n_f;
  n_i = 1.0;
  n_f = 9.7;
  t_amp = 258. + 77.0*I;
  printf("Test 'get_T'\n");
  T = get_T(t_amp, n_i, n_f);
  printf("The result of 'get_T' is %lf\n\n", T);

  printf("Test adding stuff to an array\n");
  int array[100];
  int count;
  array[0] = 42;
  count = 0;
  printf("The value in array[0] is %d\n\n", array[0]);
  
  double _Complex *** c_array_3d;
  int n;
  n = 3;
  printf("Test 'make_complex_nd' with dimension 3\n");
  c_array_3d = make_complex_nd(n);
  printf("Slice 1:\n%lf+%lfi %lf+%lfi\n%lf+%lfi %lf+%lfi\n",
	 creal(c_array_3d[0][0][0]), cimag(c_array_3d[0][0][0]),
	 creal(c_array_3d[0][0][1]), cimag(c_array_3d[0][0][1]),
	 creal(c_array_3d[0][1][0]), cimag(c_array_3d[0][1][0]),
	 creal(c_array_3d[0][1][1]), cimag(c_array_3d[0][1][1]));
  printf("Slice 2:\n%lf+%lfi %lf+%lfi\n%lf+%lfi %lf+%lfi\n",
	 creal(c_array_3d[1][0][0]), cimag(c_array_3d[1][0][0]),
	 creal(c_array_3d[1][0][1]), cimag(c_array_3d[1][0][1]),
	 creal(c_array_3d[1][1][0]), cimag(c_array_3d[1][1][0]),
	 creal(c_array_3d[1][1][1]), cimag(c_array_3d[1][1][1]));
  printf("Slice 3:\n%lf+%lfi %lf+%lfi\n%lf+%lfi %lf+%lfi\n",
	 creal(c_array_3d[2][0][0]), cimag(c_array_3d[2][0][0]),
	 creal(c_array_3d[2][0][1]), cimag(c_array_3d[2][0][1]),
	 creal(c_array_3d[2][1][0]), cimag(c_array_3d[2][1][0]),
	 creal(c_array_3d[2][1][1]), cimag(c_array_3d[2][1][1]));

  // make a mock simulation using the inputs:
  // thickness, dielectric constant, and loss tangent

  printf("\n>>>> Testing grabbing from arrays and calculating things, then putting values back into new arrays. <<<<\n\n");

  double dielectrics[5] = {1.0, 2.4, 3.15, 6.15, 9.7};
  double thicknesses[5] = {1000.0, 15.0*2.54E-5, 5.0*2.54E-5, 5.0*2.54E-5, 1000.0};
  double losses[5] = {0.0, 2.5E-4, 1.7E-3, 1.526E-3, 7.4E-4};

  double _Complex ks[5];
  double deltas[5];
  double index[5];
  // int count;
  int i;

  printf("Test converting the array of dielectric constants to refractive indices.\n");
  for (i=0; i<5; i++) {
    index[i] = get_index(dielectrics[i]);
  }
  printf("The conversion from dielectric to index is:\n");
  for (i=0; i<5; i++) {
    printf("%lf ---> %lf\n", dielectrics[i], index[i]);
  }

  printf("\nTest calculating the wavenumbers.\n");
  for (i=0; i<5; i++) {
    ks[i] = find_k(index[i], freq, losses[i]);
  }
  printf("The dimensioned wavenumber (with units m^(-1)) for each layer is:\n");
  for (i=0; i<5; i++) {
    printf("%lf+%lfi\n", creal(ks[i]), cimag(ks[i]));
  }
  for (i=0; i<5; i++) {
    deltas[i] = ks[i]*thicknesses[i];
  }
  printf("\nAnd the dimensionless wavenumber for each layer is:\n");
  for (i=0; i<5; i++) {
    printf("%lf+%lf\n", creal(deltas[i]), cimag(deltas[i]));
  }
    

  printf("\n#####################\n### END DEBUGGING ###\n#####################\n\n");
}

double get_index(double dielectric) {
  return sqrt(dielectric);
}

double _Complex find_k(double index, double freq, double loss) {
  return (2*M_PI*index*freq*(1-0.5*loss*I))/3E8;
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
  int i,j,val;
  val = 0;
  matrix = (double **)malloc(sizeof(double *)*2);
  for (i=0; i<2; i++) {
    matrix[i] = (double *)malloc(sizeof(double)*2);
    for (j=0; j<2; j++) {
      matrix[i][j] = val;
      val ++;
    }
  }
  return matrix;
}

double _Complex ** make_complex_2x2(void) {
  double _Complex ** matrix;
  int i,j,val;
  val = 0;
  matrix = (double _Complex **)malloc(sizeof(double _Complex *)*2);
  for (i=0; i<2; i++) {
    matrix[i] = (double _Complex *)malloc(sizeof(double _Complex)*2);
    for (j=0; j<2; j++) {
      matrix[i][j] = val+val*I;
      val ++;
    }
  }
  return matrix;
}

double _Complex *** make_complex_nd(int n) {
  double _Complex *** nd_matrix;
  int i,j,k,val;
  val = 0;
  nd_matrix = (double _Complex ***)malloc(sizeof(double _Complex **)*n);
  for (i=0; i<n; i++) {
    nd_matrix[i] = (double _Complex **)malloc(sizeof(double _Complex *)*2);
    for (j=0; j<2; j++) {
      nd_matrix[i][j] = (double _Complex *)malloc(sizeof(double _Complex)*2);
      for (k=0; k<2; k++) {
	nd_matrix[i][j][k] = val;
	val ++;
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
