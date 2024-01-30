
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
#include "eigen/Eigen/StdVector"
#include "eigen/Eigen/Dense"
#include "eigen/unsupported/Eigen/FFT"
#include "helper_functions.h"

using namespace std;
using namespace Eigen;

int main() {

    string ip_path = "Enter path of the imput image here";
    string op_path = "Enter path of the output image here";

    int width, height, channels;
    //Loading the input image
    unsigned char *img = stbi_load(ip_path.c_str(), &width, &height, &channels, 0);
    
    rgb op = unsignedCharArrayTo3dMatrix(img, height, width, channels);

    rgb s_op = threeDProcessing(op);

    //Converting from vector<MatrixXf> to unsigned char*
    unsigned char* s_img = Matrix3dToUnsignedCharArray(s_op, height, width, channels); 

    //Writing the smoothed image
    stbi_write_png(op_path.c_str(), width, height, channels, s_img, width*channels);

    return(0);
}    
