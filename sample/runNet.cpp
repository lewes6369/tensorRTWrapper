#include <string>
#include <sstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../code/include/TrtNet.h"
#include "argsParser.h"
#include "configs.h"

using namespace std;
using namespace argsParser;
using namespace Tn;

unique_ptr<float[]> prepareImage(const string& fileName)
{
    using namespace cv;

    Mat img = imread(fileName);
    if(img.data== nullptr)
    {
        std::cout << "can not open image :" << fileName  << std::endl;
        return std::unique_ptr<float[]>(nullptr); 
    } 

    int c = parser::getIntValue("C");
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");  

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(h,w));

    cv::Mat img_float;
    if (c == 3)
        resized.convertTo(img_float, CV_32FC3);
    else
        resized.convertTo(img_float, CV_32FC1);

    //HWC TO CHW
    cv::Mat input_channels[c];
    cv::split(img_float, input_channels);

    float * data = new float[h*w*c];
    auto result = data;
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }     

    return std::unique_ptr<float[]>(result);
}

int main( int argc, char* argv[] )
{
    parser::ADD_ARG_FLOAT("prototxt",Desc("input deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_FLOAT("caffemodel",Desc("input caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_STRING("input",Desc("input image file"),DefaultValue(INPUT_IMAGE),ValueDesc("file"));
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_INT("iterTimes",Desc("iterations"),DefaultValue(to_string(ITER_TIMES)));
 
    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);

    string deployFile = parser::getStringValue("prototxt");
    string caffemodelFile = parser::getStringValue("caffemodel");
    std::vector<std::vector<float>> calibratorData;
    trtNet net(deployFile,caffemodelFile,{"prob"},calibratorData);

    string inputImage = parser::getStringValue("input");
    auto inputData = prepareImage(inputImage);
    int outputCount = net.getOutputSize()/sizeof(float);
    std::unique_ptr<float[]> outputData(new float[outputCount]);

    for (int i = 0 ;i<ITER_TIMES;++i)
        net.doInference(inputData.get(), outputData.get());

    net.printTime();

    auto result = outputData.get();
    std::cout << "*************result************" << std::endl;
    //argmax
    auto index = std::distance(&result[0], std::max_element(&result[0], &result[outputCount] + 1));
    std::cout << "class:" << index << " " << result[index] << " " << std::endl;
    
    return 0;
}
