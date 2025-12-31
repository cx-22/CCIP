#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <unordered_map>

struct Parameter
{
    std::string label;
    double value;
    double min_value;
    double max_value;
    int decimals;
    double step;
};

struct FunctionData
{
    std::string id;
    std::string name;
    std::vector<Parameter> parameters;
};

struct FunctionState
{
    std::string id;
    std::vector<float> params;
};

using FunctionPtr = void(*)(cv::Mat&, cv::Mat&, std::vector<float>&);

extern std::unordered_map<std::string, FunctionPtr> g_func_map;
extern std::vector<FunctionData> g_function_data;

void grayscale(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void quantize(cv::Mat& in, cv::Mat& out, std::vector<float>& params);

void add(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void sub(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void divide(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void mul(cv::Mat& in, cv::Mat& out, std::vector<float>& params);

void boxBlur(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void gauBlur(cv::Mat& in, cv::Mat& out, std::vector<float>& params);

void sobelGray(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void sobelRGB(cv::Mat& in, cv::Mat& out, std::vector<float>& params);

void dogExtended(cv::Mat& in, cv::Mat& out, std::vector<float>& params);
void dogSuper(cv::Mat& in, cv::Mat& out, std::vector<float>& params);

void build_funcs();

#endif // FUNCTIONS_H
