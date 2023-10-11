#include "helper_functions.h"
#include <iostream>
#include <string>
#include <complex>
#include <vector>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <cmath>
#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/FFT"

using namespace std;
using namespace Eigen;

int main() {

    // For fixed random numbers
    srand(0);
    int start = -5, stop = 5;
    int n = stop - start;

    VectorXf ip_1d (n);
    VectorXf op_1d (n);

    //Creating the input vector
    for (int i = start; i < stop; ++i)
        ip_1d(i - start) = i;   

    //Creating the output vector
    op_1d(int(n/2)) = 0.5;
    fill(begin(op_1d) + stop+1, end(op_1d), 1.0);
    for (int i = 0; i < n; i++)
        op_1d(i) = op_1d(i) + (float)rand()/RAND_MAX;

    display(op_1d);

    VectorXf s_op_1d = oneD_processing(op_1d);

    display(s_op_1d);

    return(0);
}    