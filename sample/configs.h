#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <string>
namespace Tn
{
    //src input image size
    static const int INPUT_CHANNEL = 3;
    static const int INPUT_WIDTH = 608;
    static const int INPUT_HEIGHT = 608; 
    static const int RESIZE_H = 256;
    static const int RESIZE_W = 256;
    static const float SCALE = 0.017f;
    static const char* MEAN_VALUE = "103.94,116.78,123.68";

    //input data
    static const char* INPUT_PROTOTXT = "alexnet.prototxt";
    static const char* INPUT_CAFFEMODEL = "alexnet.caffemodel";
    static const std::string INPUT_IMAGE = "test.jpg";
    static const char* EVAL_LIST = "";
    static const char* CALIBRATION_LIST = "";
    static const char* MODE = "fp32";
    static const char* OUTPUTS = "prob";

    static const int ITER_TIMES = 1000;
}
#endif