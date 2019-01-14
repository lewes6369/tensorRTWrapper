# TRTWrapper

### Desc
    a wrapper for tensorRT net (parser caffe)
### Test Environments
    Ubuntu  16.04
    TensorRT 5.0.2.6/4.0.1.6
    CUDA 9.2
### About Wraper
you can use the wrapper like this:
```cpp
//normal
std::vector<std::vector<float>> calibratorData;
trtNet net("vgg16.prototxt","vgg16.caffemodel",{"prob"},calibratorData);
//fp16
trtNet net_fp16("vgg16.prototxt","vgg16.caffemodel",{"prob"},calibratorData,RUN_MODE:FLOAT16);
//int8
trtNet net_int8("vgg16.prototxt","vgg16.caffemodel",{"prob"},calibratorData,RUN_MODE:INT8);

//run inference:
net.doInference(input_data.get(), outputData.get());

//can print time cost
net.printTime();

//can write to engine and load From engine
net.saveEngine("save_1.engine");
trtNet net2("save_1.engine");
```
when you need add new plugin ,just add the plugin code to pluginFactory
### Run Sample
```bash
#for classification
cd sample
mkdir build
cd build && cmake .. && make && make install
cd ..
./install/runNet --caffemodel=${CAFFE_MODEL_NAME} --prototxt=${CAFFE_PROTOTXT} --input=./test.jpg
```
