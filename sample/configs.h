#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <string>
namespace Tn
{
    const int INPUT_CHANNEL = 3;
    const std::string INPUT_PROTOTXT ="bvlc_alexnet.prototxt";
    const std::string INPUT_CAFFEMODEL = "bvlc_alexnet.caffemodel";
    const std::string INPUT_IMAGE = "test.jpg";
    const int INPUT_WIDTH = 416;
    const int INPUT_HEIGHT = 416;
    const int ITER_TIMES =1000;
}

#endif