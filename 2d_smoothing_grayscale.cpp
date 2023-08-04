#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <complex>
#include <vector>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <cmath>
#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/FFT"

#define STB_IMAGE_IMPLEMENTATION
#include "stb-master/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb-master/stb_image_write.h"

using namespace std;
using namespace Eigen;

//Function for computing 2D FFT
MatrixXcf fft_func(MatrixXf image, int rows, int cols) {
    FFT<float> fft;
    MatrixXcf spectrum(rows, cols);

    // FFT along each row
    for (int i = 0; i < rows; ++i) {
        VectorXf row = image.row(i);
        VectorXcf row_spectrum(cols);
        row_spectrum = fft.fwd(row);
        spectrum.row(i) = row_spectrum;
    }

    // FFT along each column
    for (int j = 0; j < cols; ++j) {
        VectorXcf col = spectrum.col(j);
        VectorXcf col_spectrum(rows);
        col_spectrum = fft.fwd(col);
        spectrum.col(j) = col_spectrum;
    }
    return(spectrum);
}

//Function for computing 2D IFFT
MatrixXcf ifft_func(MatrixXcf spectrum, int rows, int cols) {
    FFT<float> fft;
    MatrixXcf image(rows, cols);

    // IFFT along each column
    for (int j = 0; j < cols; ++j) {
        VectorXcf col = spectrum.col(j);
        VectorXcf col_image(rows);
        col_image = fft.inv(col);
        image.col(j) = col_image;
    }

    // IFFT along each row
    for (int i = 0; i < rows; ++i) {
        VectorXcf row = image.row(i);
        VectorXcf row_image(cols);
        row_image = fft.inv(row);
        image.row(i) = row_image;
    }
    return(image);
}

int main() {

    string ip_path = "C:/Users/asrivast/source/repos/L0Minimization/inputs/1_gray.png";
    string op_path = "C:/Users/asrivast/source/repos/L0Minimization/outputs/1_gray.png";

    int width, height, channels;
    //Loading the input image
    unsigned char *img = stbi_load(ip_path.c_str(), &width, &height, &channels, 0);
    
    MatrixXf op(height, width);
    //Converting from unsigned char* to MatrixXf
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            op(i, j) = static_cast<float>(img[i * width + j])/255;

    MatrixXf fx(height, width);
    MatrixXf fy(height, width);
    MatrixXcf ones(height, width);
    FFT<float> fft;

    ones.setConstant(1.0);
    fx.setConstant(0.0);
    fy.setConstant(0.0);

    fx(0,0) = 1.0;
    fx(0, width-1) = -1.0;  

    fy(0,0) = 1.0;
    fy(height-1,0) = -1.0;

    //Parameters
    float lambdu = 0.02;
    float beta_0 = 2*lambdu;
    float beta_max = 10000.0;
    float kappa = 2.0;

    MatrixXf s_op = op;
    float beta = beta_0;
    int i=0;

    //Algorithm 1 of the paper
    while(beta<beta_max) {
        i = i+1;
        
        MatrixXf h(height, width);
        MatrixXf v(height, width);

        h.setConstant(0.0);
        v.setConstant(0.0);

        MatrixXf grad_x(height, width);
        MatrixXf grad_y(height, width);

        grad_x.setConstant(0.0);
        grad_y.setConstant(0.0);

        MatrixXf s_op_shift_x(height, width);
        MatrixXf s_op_shift_y(height, width);

        for(int i=0; i<height;++i) {
            s_op_shift_x.row(i).segment(0, width - 1) = s_op.row(i).segment(1, width - 1);
            s_op_shift_x.row(i)(width - 1) = s_op(i,0);
        }    

        for(int j=0; j<width;++j) {
            s_op_shift_y.col(j).segment(0, height - 1) = s_op.col(j).segment(1, height - 1);
            s_op_shift_y.col(j)(height - 1) = s_op(0,j);
        }
        
        //Calculating horizontal and vertical L0 gradient
        grad_x = s_op - s_op_shift_x;
        grad_y = s_op - s_op_shift_y;

        //Equation 12 of the paper
        h = ((grad_x.array().square() + grad_y.array().square()) > lambdu/beta).select(grad_x,0.0);
        v = ((grad_x.array().square() + grad_y.array().square()) > lambdu/beta).select(grad_y,0.0);

        MatrixXcf fft_op = fft_func(op, height, width);
        MatrixXcf fft_h = fft_func(h, height, width);
        MatrixXcf fft_v = fft_func(v, height, width);
        MatrixXcf fft_fx = fft_func(fx, height, width);
        MatrixXcf fft_fx_cc = fft_fx.conjugate();
        MatrixXcf fft_fy = fft_func(fy, height, width);
        MatrixXcf fft_fy_cc = fft_fy.conjugate();

        MatrixXcf temp1 = (((fft_h.array())*(fft_fx_cc.array()) + (fft_v.array())*(fft_fy_cc.array()))*(beta)).matrix();

        MatrixXcf num = fft_op + temp1;

        MatrixXcf temp2 = (((fft_fx.array())*(fft_fx_cc.array()) + (fft_fy.array())*(fft_fy_cc.array()))*(beta)).matrix();

        MatrixXcf den = ones + temp2;

        MatrixXcf res = (num.array()/den.array()).matrix();
        //Equation 8 of the paper
        s_op = (ifft_func(res, height, width)).real();

        beta = beta * kappa;
        
    }

    //Converting from MatrixXf to unsigned char*
    unsigned char* s_img = new unsigned char[height * width];
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
        s_img[i * width + j] = static_cast<unsigned char>(s_op(i, j) * 255.0f);
    //Writing the smoothed image
    stbi_write_png(op_path.c_str(), width, height, channels, s_img, width*channels);

    return(0);
}    