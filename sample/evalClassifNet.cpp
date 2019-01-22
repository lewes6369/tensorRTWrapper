#include <string>
#include <sstream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../code/include/TrtNet.h"
#include "argsParser.h"
#include "dataReader.h"
#include "eval.h"
#include "configs.h"

using namespace std;
using namespace argsParser;
using namespace Tn;

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
}

cv::Mat GetMeanMat()
{
    using namespace cv;
    static std::unique_ptr<Mat> MeanMat = nullptr;
    if (MeanMat.get() != nullptr)
        return *MeanMat;

    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");  
    int c = parser::getIntValue("C");
    
    string meanStr = parser::getStringValue("meanValue");
    auto meansValues = split(meanStr,',');
    float scale = parser::getFloatValue("scale");  
    
    assert(meansValues.size() == c);
    vector<Mat> means(c);
    for (int i = 0 ;i<c ;++i)
        means[i] = Mat(h, w, CV_32FC1, std::stof(meansValues[i]) * scale);

    MeanMat.reset(new Mat(h, w, CV_32FC3));
    cv::merge(means.data(), c, *MeanMat);

    return *MeanMat;
}


vector<float> preprocess(const string& fileName)
{
    using namespace cv;

    Mat img = imread(fileName);

    if(img.data== nullptr)
    {
        std::cout << "can not open image :" << fileName  << std::endl;
        return {}; 
    } 

    int channel = parser::getIntValue("C");
    
    //channel 
    Mat sample;
    if (img.channels() == 3 && channel == 1)
        cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && channel == 1)
        cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && channel == 3)
        cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && channel == 3)
        cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    //resize
    int r_h = parser::getIntValue("RH");
    int r_w = parser::getIntValue("RW"); 
    cv::Mat resized;
    cv::resize(sample, resized, cv::Size(r_h,r_w));

    //crop
    int h = parser::getIntValue("H");
    int w = parser::getIntValue("W");  
    int h_off = 0;
    int w_off = 0;
    h_off = (r_h - h) / 2;
    w_off = (r_w - w) / 2;
    Rect roi(w_off, h_off, w, h);
    Mat croppedImg = resized(roi);

    //to float and scale
    cv::Mat img_float;
    float scale = parser::getFloatValue("scale");  
    if (channel == 3)
        croppedImg.convertTo(img_float, CV_32FC3, scale);
    else
        croppedImg.convertTo(img_float, CV_32FC1, scale);


    // //mean mat
    auto meanFile = GetMeanMat();
    Mat subMeanImg;
    cv::subtract(img_float, meanFile, subMeanImg);

    //HWC TO CHW
    vector<Mat> input_channels(channel);
    cv::split(subMeanImg, input_channels.data());

    vector<float> result(h*w*channel);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < channel; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

int main( int argc, char* argv[] )
{
    parser::ADD_ARG_INT("C",Desc("channel"),DefaultValue(to_string(INPUT_CHANNEL)));
    parser::ADD_ARG_INT("H",Desc("height"),DefaultValue(to_string(INPUT_HEIGHT)));
    parser::ADD_ARG_INT("W",Desc("width"),DefaultValue(to_string(INPUT_WIDTH)));
    parser::ADD_ARG_INT("RH",Desc("image process resized Height"),DefaultValue(to_string(RESIZE_H)));
    parser::ADD_ARG_INT("RW",Desc("image process resized Width"),DefaultValue(to_string(RESIZE_W)));
    parser::ADD_ARG_FLOAT("scale",Desc("image process scale"),DefaultValue(to_string(SCALE)));
    parser::ADD_ARG_STRING("meanValue",Desc("image mean value before scale"),DefaultValue(MEAN_VALUE));
    
    parser::ADD_ARG_STRING("caffemodel",Desc("input caffemodel"),DefaultValue(INPUT_CAFFEMODEL),ValueDesc("file"));
    parser::ADD_ARG_STRING("prototxt",Desc("input deploy"),DefaultValue(INPUT_PROTOTXT),ValueDesc("file"));
    parser::ADD_ARG_STRING("evallist",Desc("load test files from list"),DefaultValue(EVAL_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("calib",Desc("load calibration files from list"),DefaultValue(CALIBRATION_LIST),ValueDesc("file"));
    parser::ADD_ARG_STRING("outputs",Desc("output nodes name"),DefaultValue(OUTPUTS));
    parser::ADD_ARG_STRING("mode",Desc("runtime mode"),DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));

    if(argc < 2){
        parser::printDesc();
        exit(-1);
    }

    parser::parseArgs(argc,argv);

    vector<vector<float>> calibData;
    string calibFileList = parser::getStringValue("calib");
    string mode = parser::getStringValue("mode");
    if(calibFileList.length() > 0 && mode == "int8")
    {   
        cout << "find calibration file,loading ..." << endl;
      
        ifstream file(calibFileList);  
        if(!file.is_open())
        {
            cout << "read file list error,please check file :" << calibFileList << endl;
            exit(-1);
        }

        string strLine;  
        while( getline(file,strLine) )                               
        { 
            //std::cout << strLine << std::endl;
            auto data = preprocess(strLine);
            calibData.emplace_back(data);
        } 
        file.close();
    }
    
    string deployFile = parser::getStringValue("prototxt");
    string caffemodelFile = parser::getStringValue("caffemodel");
    string outputNodes = parser::getStringValue("outputs");
    auto outputNames = split(outputNodes,',');

    trtNet net(deployFile,caffemodelFile,outputNames,calibData);

    int outputCount = net.getOutputSize()/sizeof(float);
    std::unique_ptr<float[]> outputData(new float[outputCount]);

    list<vector<float>> outputs;
    list<int> groundTruth;
    string listFile = parser::getStringValue("evallist");

    cout << "loading process list from " << listFile << endl;
    list<Source> inputs = readLabelFileList(listFile);

    int tp1 = 0,fp1 =0;
    int tp5 = 0,fp5 =0;
    const int printInterval = 500;
    int i = 0;

    for (const auto& source :inputs)
    {

        std::cout << "process: " << source.fileName << std::endl;
        vector<float> inputData = preprocess(source.fileName);
        if (!inputData.data())
            continue;

        net.doInference(inputData.data(), outputData.get());

        //Get Output    
        auto output = outputData.get();
        
        vector<float> res(output,&output[outputCount]);
        outputs.emplace_back(res);
        groundTruth.push_back(source.label);

        if(++i % printInterval == 0)
        {
            evalTopResult(outputs,groundTruth,&tp1,&fp1,1);
            evalTopResult(outputs,groundTruth,&tp5,&fp5,5); 

            outputs.clear();
            groundTruth.clear();
        }
    }

    evalTopResult(outputs,groundTruth,&tp1,&fp1,1);
    evalTopResult(outputs,groundTruth,&tp5,&fp5,5); 

    net.printTime();

    return 0;
}

