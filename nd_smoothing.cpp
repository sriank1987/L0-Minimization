#include "helper_functions.h"
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

using namespace std;
using namespace Eigen;
typedef vector<MatrixXf> rgb;

int main() {

    string ip_path = "Enter path of the imput image here";
    string op_path = "Enter path of the output image here";

    vector<int> dims = {876, 584};

    int size = 1;
    for(int i=0;i<dims.size();i++) {
        size = size*dims[i];
    }

    VectorXf input(size);
    input.setConstant(1.0);

    if(dims.size() == 1){
        VectorXf s_op_1d = oneD_processing(input);
        display(s_op_1d);
    }
    
    else if (dims.size() == 2){
        unsigned char *img = vectorXfToUnsignedCharArray(input);
        MatrixXf op = unsignedCharArrayTo2dMatrix(img,dims[0],dims[1]);
        MatrixXf s_op = twoD_processing(op);
        unsigned char* s_img = Matrix2dToUnsignedCharArray(s_op,dims[0],dims[1]);
        stbi_write_png(op_path.c_str(), dims[1], dims[0], 1, s_img, dims[1]*1);
        //VectorXf s_op_2d = unsignedCharArrayToVectorXf(s_img, size);
    }
    else if (dims.size() == 3){
        unsigned char *img = vectorXfToUnsignedCharArray(input);
        rgb op = unsignedCharArrayTo3dMatrix(img, dims[0], dims[1], dims[2]);
        rgb s_op = threeDProcessing(op);
        unsigned char* s_img = Matrix3dToUnsignedCharArray(s_op,dims[0],dims[1], dims[2]);
        stbi_write_png(op_path.c_str(), dims[1], dims[0], dims[2], s_img, dims[1]*dims[2]);
        //VectorXf s_op_3d = unsignedCharArrayToVectorXf(s_img, size);
    }
    else cout <<"Code unavailable for higher dimensions";
    
    return(0);
}    
