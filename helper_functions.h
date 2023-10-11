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

#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

void display(Eigen::VectorXf c);
Eigen::VectorXf oneD_processing(Eigen::VectorXf vec);
Eigen::MatrixXcf fft_func(Eigen::MatrixXf image, int rows, int cols);
Eigen::MatrixXcf ifft_func(Eigen::MatrixXcf spectrum, int rows, int cols);
Eigen::MatrixXf twoD_processing(Eigen::MatrixXf vec);
unsigned char* vectorXfToUnsignedCharArray(const Eigen::VectorXf& vectorData);
Eigen::MatrixXf unsignedCharArrayTo2dMatrix(const unsigned char* img, int height, int width);
unsigned char* Matrix2dToUnsignedCharArray(const Eigen::MatrixXf s_op, int height, int width);
Eigen::VectorXf unsignedCharArrayToVectorXf(const unsigned char* ucharArray, int numElements);
rgb unsignedCharArrayTo3dMatrix(const unsigned char* img, int height, int width, int channels);
unsigned char* Matrix3dToUnsignedCharArray(const rgb s_op, int height, int width, int channels);
rgb threeDProcessing (rgb op);
#endif