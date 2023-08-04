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

// Displaying the vector
void display(VectorXf& c ) {
    for (int i = 0; i < c.size(); ++i)
        cout << c(i) << " ";
    cout << endl;
}

int main() {

    // For fixed random numbers
    srand(0);
    int start = -5, stop = 5;
    int n = stop - start;

    VectorXf ip_1d (n);
    VectorXf op_1d (n);
    VectorXf fx (n);
    VectorXf ones (n);
    FFT<float> fft;

    ones.setConstant(1.0);
    fx.setConstant(0.0);
    op_1d.setConstant(0.0);

    //fx = {1,-1}
    fx(0) = 1.0;
    fx(n-1) = -1.0;

    //Creating the input vector
    for (int i = start; i < stop; ++i)
        ip_1d(i - start) = i;   

    //Creating the output vector
    op_1d(int(n/2)) = 0.5;
    fill(begin(op_1d) + stop+1, end(op_1d), 1.0);
    for (int i = 0; i < n; i++)
        op_1d(i) = op_1d(i) + (float)rand()/RAND_MAX;

    display(op_1d);

    //Parameters
    float lambdu = 0.02;
    float beta_0 = 2*lambdu;
    float beta_max = 10000.0;
    float kappa = 2.0;

    VectorXf s_op_1d = op_1d;
    float beta = beta_0;
    int i=0;

    //Algorithm 1 of the paper
    while(beta<beta_max) {
        i = i+1;
        
        VectorXf h (n);
        VectorXf grad (n);

        VectorXf s_op_1d_1shift(n);
        s_op_1d_1shift.segment(0, n - 1) = s_op_1d.segment(1, n - 1);
        s_op_1d_1shift(n - 1) = s_op_1d(0);
        
        //Implementing L0 gradient
        grad = s_op_1d - s_op_1d_1shift;

        //Equation 12 of paper
        h = (grad.array().square() > lambdu/beta).select(grad,0.0);

        VectorXcf fft_op_1d = fft.fwd(op_1d);
        VectorXcf fft_h = fft.fwd(h);
        VectorXcf fft_fx = fft.fwd(fx);
        VectorXcf fft_fx_cc = fft_fx.conjugate();

        VectorXcf temp1 = ((fft_h.array())*(fft_fx_cc.array())*(beta)).matrix();

        VectorXcf num = fft_op_1d + temp1;

        VectorXcf temp2 = ((fft_fx.array())*(fft_fx_cc.array())*(beta)).matrix();

        VectorXcf den = ones + temp2;

        VectorXcf res = (num.array()/den.array()).matrix();

        //Equation 8 of the paper
        s_op_1d = (fft.inv(res)).real();

        beta = beta * kappa;
        
    }

    display(s_op_1d);

    return(0);
}    
