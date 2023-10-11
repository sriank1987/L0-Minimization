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

#define STB_IMAGE_IMPLEMENTATION
#include "stb-master/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb-master/stb_image_write.h"

using namespace std;
using namespace Eigen;
typedef vector<MatrixXf> rgb;

void display(Eigen::VectorXf c) {
    for (int i = 0; i < c.size(); ++i)
        cout << c[i] << " ";
    cout << endl;
}

Eigen::VectorXf oneD_processing(Eigen::VectorXf vec){
    int n = vec.size();

    VectorXf fx (n);
    VectorXf ones (n);
    FFT<float> fft;

    ones.setConstant(1.0);
    fx.setConstant(0.0);

    //fx = {1,-1}
    fx(0) = 1.0;
    fx(n-1) = -1.0;
    //Parameters
    float lambdu = 0.02;
    float beta_0 = 2*lambdu;
    float beta_max = 10000.0;
    float kappa = 2.0;

    VectorXf s_op_1d = vec;
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

        VectorXcf fft_op_1d = fft.fwd(vec);
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
    return(s_op_1d);
}

//Function for computing 2D FFT
Eigen::MatrixXcf fft_func(Eigen::MatrixXf image, int rows, int cols) {
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
Eigen::MatrixXcf ifft_func(Eigen::MatrixXcf spectrum, int rows, int cols) {
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

Eigen::MatrixXf twoD_processing(Eigen::MatrixXf mat){
    
    int height = mat.rows();
    int width = mat.cols();

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

    MatrixXf s_op = mat;
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

        MatrixXcf fft_op = fft_func(mat, height, width);
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
    return(s_op);
}

unsigned char* vectorXfToUnsignedCharArray(const Eigen::VectorXf& vectorData) {
    // Determine the size required for the unsigned char array (assuming 4 bytes per float)
    size_t size = vectorData.size() * sizeof(float);
    
    // Allocate memory for the unsigned char array
    unsigned char* ucharArray = new unsigned char[size];

    // Copy the float data into the unsigned char array
    memcpy(ucharArray, vectorData.data(), size);

    return ucharArray;
}

Eigen::MatrixXf unsignedCharArrayTo2dMatrix(const unsigned char* img, int height, int width) {
    MatrixXf op(height, width);
    //Converting from unsigned char* to MatrixXf
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            op(i, j) = static_cast<float>(img[i * width + j])/255;

    return(op);        
}

unsigned char* Matrix2dToUnsignedCharArray(const Eigen::MatrixXf s_op, int height, int width) {
    //Converting from MatrixXf to unsigned char*
    unsigned char* s_img = new unsigned char[height * width];
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
        s_img[i * width + j] = static_cast<unsigned char>(s_op(i, j) * 255.0f);
    
    return(s_img);
}

Eigen::VectorXf unsignedCharArrayToVectorXf(const unsigned char* ucharArray, int numElements) {
    Eigen::VectorXf vectorData(numElements);

    // Copy the data from the unsigned char* array to the Eigen::VectorXf
    std::memcpy(vectorData.data(), ucharArray, numElements * sizeof(float));

    return vectorData;
}

rgb unsignedCharArrayTo3dMatrix(const unsigned char* img, int height, int width, int channels) {
    rgb op(channels, MatrixXf(height, width));
    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                // Extract the corresponding RGB pixel from the image data
                int pixelIndex = (i * width + j) * channels + c;
                // Scale the unsigned char value to the desired range (e.g., [0, 1]) and map it to the matrix
                op[c](i, j) = static_cast<float>(img[pixelIndex]) / 255.0f;
            }
        }
    }    
    return (op);    
}

unsigned char* Matrix3dToUnsignedCharArray(const rgb s_op, int height, int width, int channels) {
    unsigned char* s_img = new unsigned char[height * width * channels];
    int index = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int c = 0; c < channels; ++c) {
                // Scale the floating-point value to [0, 255] and cast it to unsigned char
                s_img[index++] = static_cast<unsigned char>(s_op[c](i, j) * 255.0f);
            }
        }
    }
    return (s_img);
}

rgb threeDProcessing (rgb op){
    
    int channels = op.size();
    int height = op[0].rows();
    int width = op[0].cols();

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
    float lambdu = 0.04;
    float beta_0 = 2*lambdu;
    float beta_max = 10000.0;
    float kappa = 2.0;

    rgb s_op = op;
    float beta = beta_0;
    int i=0;

    while(beta<beta_max) {
        i = i+1;
        //Doing for every image channel of the image
        for (int c=0; c<channels;++c) {
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
                s_op_shift_x.row(i).segment(0, width - 1) = s_op[c].row(i).segment(1, width - 1);
                s_op_shift_x.row(i)(width - 1) = s_op[c](i,0);
            }    

            for(int j=0; j<width;++j) {
                s_op_shift_y.col(j).segment(0, height - 1) = s_op[c].col(j).segment(1, height - 1);
                s_op_shift_y.col(j)(height - 1) = s_op[c](0,j);
            }
            //Calculating horizontal and vertical L0 gradient
            grad_x = s_op[c] - s_op_shift_x;
            grad_y = s_op[c] - s_op_shift_y;

            h = ((grad_x.array().square() + grad_y.array().square()) > lambdu/beta).select(grad_x,0.0);
            v = ((grad_x.array().square() + grad_y.array().square()) > lambdu/beta).select(grad_y,0.0);

            MatrixXcf fft_op = fft_func(op[c], height, width);
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
            s_op[c] = (ifft_func(res, height, width)).real();
        }
        beta = beta * kappa;      
    }
    return (s_op);
}