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
#include "helper_functions.h"

using namespace std;
using namespace Eigen;

int main() {

    string ip_path = "Enter input path of the image here";
    string op_path = "Enter output path of the image here";

    int width, height, channels;
    //Loading the input image
    unsigned char *img = stbi_load(ip_path.c_str(), &width, &height, &channels, 0);
    
    MatrixXf op = unsignedCharArrayTo2dMatrix(img, height, width);

    MatrixXf s_op = twoD_processing(op);

    unsigned char* s_img = Matrix2dToUnsignedCharArray(s_op, height, width);
    //Writing the smoothed image
    stbi_write_png(op_path.c_str(), width, height, channels, s_img, width*channels);

    return(0);
}    
