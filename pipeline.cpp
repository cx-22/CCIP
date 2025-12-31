#include "pipeline.h"

Pipeline::Pipeline() {}

extern std::unordered_map<std::string, void(*)(cv::Mat&, cv::Mat&, std::vector<float>&)> g_func_map;

void Pipeline::setImage(cv::Mat img){
    current_image = img;
    viewer->display(current_image);
}

void Pipeline::processImage(std::vector<FunctionState> states){
    cv::Mat buffer1 = current_image.clone();
    cv::Mat buffer2;

    for(size_t i = 0; i < states.size(); i++){
        buffer2 = buffer1.clone();
        void(*func)(cv::Mat&, cv::Mat&, std::vector<float>&) = g_func_map.find(states[i].id)->second;
        func(buffer2, buffer1, states[i].params);
    }

    final_image = buffer1.clone();
    viewer->display(final_image);
}
