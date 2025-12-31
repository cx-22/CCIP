#include "functions.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>


std::unordered_map<std::string, FunctionPtr> g_func_map;
std::vector<FunctionData> g_function_data;


void grayscale(cv::Mat& in, cv::Mat& out, std::vector<float>&)
{
    cv::cvtColor(in, out, cv::COLOR_BGR2GRAY);
}

void quantize(cv::Mat& in, cv::Mat& out, std::vector<float>& params)
{
    int val = static_cast<int>(params[0]);
    cv::Mat table(1, 256, CV_8U);

    for (int i = 0; i < 256; i++){
        table.at<uchar>(0, i) = static_cast<uchar>(val * (i / val));
    }

    int rows = in.rows;
    int cols = in.cols * in.channels();
    uchar* ptr_in;
    uchar* ptr_out;

    for (int x = 0; x < rows; x++)
    {
        ptr_in = in.ptr<uchar>(x);
        ptr_out = out.ptr<uchar>(x);

        for (int y = 0; y < cols; y++){
            uchar num = table.at<uchar>(0, ptr_in[y]);
            if (num < 0) {num = 0;}
            if (num > 255.0f) {num = 255.0f;}
            ptr_out[y] = static_cast<uchar>(num);
        }
    }
}

void add(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    out = in + cv::Scalar(params[0], params[0], params[0]);
}

void sub(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    out = in - cv::Scalar(params[0], params[0], params[0]);
}

void divide(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    out = in / cv::Scalar(params[0], params[0], params[0]);
}

void mul(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    cv::multiply(in, cv::Scalar(params[0], params[0], params[0]), out);
}

void boxBlur(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    int size = static_cast<int>(params[0]);
    float val = 1.0f / (params[0] * params[0]);

    cv::Mat kernel(size, size, CV_32F);
    kernel.setTo(val);
    cv::filter2D(in, out, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
}

void gauBlur(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    double sigma = static_cast<double>(params[0]);

    cv::GaussianBlur(in, out, cv::Size(0, 0), sigma);
}

void sobelGray(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    cv::Mat sobel_x, sobel_y;
    cv::Mat gray;
    cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                                               -2, 0, 2,
                                               -1, 0, 1);

    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << 1, 2, 1,
                                                 0, 0, 0,
                                                 -1, -2, -1);

    cv::filter2D(gray, sobel_x, CV_32F, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gray, sobel_y, CV_32F, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat temp = sobel_x.mul(sobel_x) + sobel_y.mul(sobel_y);
    cv::sqrt(temp, out);

    if (params[0] != -1.0f){
        temp = out.clone();
        double thres = static_cast<double>(params[0]);
        if (params[1] == 0){
            cv::threshold(temp, out, thres, 255, cv::THRESH_TOZERO);
        } else if (params[1] == 1){
            cv::threshold(temp, out, thres, 255, cv::THRESH_BINARY);
        } else if (params[1] == 2){
            cv::threshold(temp, out, thres, 255, cv::THRESH_TRUNC);
        }
    }

    out.convertTo(out, CV_8U);
}

void sobelRGB(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    cv::Mat sobel_x, sobel_y;
    cv::Mat gray;
    cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1);

    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << 1, 2, 1,
                        0, 0, 0,
                        -1, -2, -1);

    cv::filter2D(gray, sobel_x, CV_32F, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gray, sobel_y, CV_32F, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    cv::Mat angle, magnitude;
    cv::phase(sobel_x, sobel_y, angle, false);
    cv::magnitude(sobel_x, sobel_y, magnitude);

    cv::Mat val;
    cv::normalize(magnitude, val, 0, 255, cv::NORM_MINMAX, CV_8U);

    if (params[0] != -1.0f){
        magnitude = val.clone();
        double thres = static_cast<double>(params[0]);

        if (params[1] == 0){
            cv::threshold(magnitude, val, thres, 255, cv::THRESH_TOZERO);
        } else if (params[1] == 1){
            cv::threshold(magnitude, val, thres, 255, cv::THRESH_BINARY);
        } else if (params[1] == 2){
            cv::threshold(magnitude, val, thres, 255, cv::THRESH_TRUNC);
        }
    }

    cv::Mat hue;
    hue = angle * (180.0 / CV_PI);
    cv::Mat mask = hue > 180;
    cv::subtract(hue, 180, hue, mask);

    hue.convertTo(hue, CV_8U);

    cv::Mat sat(val.size(), CV_8U, cv::Scalar(255));

    std::vector<cv::Mat> channels = {hue, sat, val};
    cv::Mat hsv;
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, out, cv::COLOR_HSV2BGR);

}

void dogExtended(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    double sigma1 = static_cast<double>(params[0]);
    double k = static_cast<double>(params[1]);
    float thres = static_cast<float>(params[2]);
    float tau = static_cast<float>(params[3]);
    float phi = static_cast<float>(params[4]);

    cv::Mat gau1, gau2, diff;

    cv::Mat gray_f;
    cv::cvtColor(in, gray_f, cv::COLOR_BGR2GRAY);
    gray_f.convertTo(gray_f, CV_32F, 1.0 / 255.0);

    cv::GaussianBlur(gray_f, gau1, cv::Size(0, 0), sigma1);
    cv::GaussianBlur(gray_f, gau2, cv::Size(0, 0), (sigma1 * k));

    cv::multiply(gau1, cv::Scalar(tau + 1.0f, tau + 1.0f, tau + 1.0f), gau1);
    cv::multiply(gau2, cv::Scalar(tau, tau, tau), gau2);
    cv::subtract(gau1, gau2, diff);

    cv::Mat diff_norm;
    cv::normalize(diff, diff_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    out.create(diff_norm.size(), CV_32FC1);

    int rows = diff_norm.rows;
    int cols = diff_norm.cols * diff_norm.channels();
    float* ptr_in;
    float* ptr_out;

    for (int x = 0; x < rows; x++)
    {
        ptr_in = diff_norm.ptr<float>(x);
        ptr_out = out.ptr<float>(x);

        for (int y = 0; y < cols; y++){
            if (ptr_in[y] >= thres){
                ptr_out[y] = 1.0f;
            } else {
                if (phi == -1.0f){
                    ptr_out[y] = 0;
                } else {
                    ptr_out[y] = 1.0f + std::tanh(phi * (ptr_in[y] - thres));
                }
            }
        }
    }

    cv::Mat dis;
    out.convertTo(dis, CV_8U, 255.0);
    out = dis.clone();
}

void dogSuper(cv::Mat& in, cv::Mat& out, std::vector<float>& params){

    float sigma_e = static_cast<float>(params[0]);
    double k = static_cast<double>(params[1]);
    float thres = static_cast<float>(params[2]);
    float tau = static_cast<float>(params[3]);
    float phi = static_cast<float>(params[4]);
    float sigma_c = static_cast<float>(params[5]);
    float sigma_m = static_cast<float>(params[6]);
    float sigma_a = static_cast<float>(params[7]);
    //*
    cv::Mat gray;
    cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                                                -2, 0, 2,
                                                -1, 0, 1);

    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << 1, 2, 1,
                                                0, 0, 0,
                                                -1, -2, -1);


    cv::Mat imgDiffX, imgDiffY;
    cv::filter2D(gray, imgDiffX, CV_32F, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::filter2D(gray, imgDiffY, CV_32F, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

    //std::cerr << "h\n";

    cv::Mat imgDiffXX, imgDiffYY, imgDiffXY;
    cv::multiply(imgDiffX, imgDiffY, imgDiffXY);
    cv::multiply(imgDiffX, imgDiffX, imgDiffXX);
    cv::multiply(imgDiffY, imgDiffY, imgDiffYY);

    cv::Mat J11, J22, J12;
    cv::GaussianBlur(imgDiffXX, J11, cv::Size(0, 0), sigma_c);
    cv::GaussianBlur(imgDiffXY, J12, cv::Size(0, 0), sigma_c);
    cv::GaussianBlur(imgDiffYY, J22, cv::Size(0, 0), sigma_c);

    cv::Mat tmp1, tmp2, tmp3, tmp4;
    tmp1 = J11 + J22;
    tmp2 = J11 - J22;
    multiply(tmp2, tmp2, tmp2);
    multiply(J12, J12, tmp3);
    sqrt(tmp2 + 4.0 * tmp3, tmp4);

    cv::Mat lambda1, lambda2;
    lambda1 = tmp1 + tmp4;
    lambda1 = 0.5*lambda1;
    lambda2 = tmp1 - tmp4;
    lambda2 = 0.5*lambda2;

    cv::Mat v_x = lambda1 - J11;
    cv::Mat v_y = -J12;

    cv::Mat mag;
    cv::magnitude(v_x, v_y, mag);

    cv::Mat tx = v_x / (mag + 1e-6f);
    cv::Mat ty = v_y / (mag + 1e-6f);

    cv::Mat nx =  ty;
    cv::Mat ny = -tx;

    int radius = int(std::ceil(2 * sigma_e));

    cv::Mat blur1(gray.size(), CV_32F, 0.0f);
    cv::Mat blur2(gray.size(), CV_32F, 0.0f);
    cv::Mat w1(gray.size(), CV_32F, 0.0f);
    cv::Mat w2(gray.size(), CV_32F, 0.0f);

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {

            float nxp = nx.at<float>(y,x);
            float nyp = ny.at<float>(y,x);

            for (int d = -radius; d <= radius; ++d) {
                float wE  = std::exp(-(d*d)/(2*sigma_e*sigma_e));
                float wEk = std::exp(-(d*d)/(2*(sigma_e * k)*(sigma_e * k)));

                float sx = x + d * nxp;
                float sy = y + d * nyp;

                if (sx < 0 || sy < 0 || sx >= gray.cols-1 || sy >= gray.rows-1)
                    continue;

                float val = gray.at<uchar>(int(sy), int(sx));

                blur1.at<float>(y,x) += wE  * val;
                blur2.at<float>(y,x) += wEk * val;
                w1.at<float>(y,x)    += wE;
                w2.at<float>(y,x)    += wEk;
            }
        }
    }

    blur1 /= w1;
    blur2 /= w2;



    cv::Mat diff_norm, diff;

    cv::multiply(blur1, cv::Scalar(tau + 1.0f, tau + 1.0f, tau + 1.0f), blur1);
    cv::multiply(blur2, cv::Scalar(tau, tau, tau), blur2);
    cv::subtract(blur1, blur2, diff);

    cv::normalize(diff, diff_norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    cv::Mat lic(gray.size(), CV_32F, 0.0f);
    cv::Mat w(gray.size(), CV_32F, 0.0f);

    int radiusM = int(2 * sigma_m);

    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {

            float cx = x;
            float cy = y;

            // forward
            for (int d = 1; d <= radiusM; ++d) {
                float txp = tx.at<float>(int(cy), int(cx));
                float typ = ty.at<float>(int(cy), int(cx));

                cx += txp;
                cy += typ;

                if (cx < 0 || cy < 0 || cx >= gray.cols-1 || cy >= gray.rows-1)
                    break;

                float g = std::exp(-(d*d)/(2*sigma_m*sigma_m));
                lic.at<float>(y,x) += g * diff_norm.at<float>(int(cy), int(cx));
                w.at<float>(y,x)   += g;
            }

            // backward
            cx = x;
            cy = y;
            for (int d = 1; d <= radiusM; ++d) {
                float txp = tx.at<float>(int(cy), int(cx));
                float typ = ty.at<float>(int(cy), int(cx));

                cx -= txp;
                cy -= typ;

                if (cx < 0 || cy < 0 || cx >= gray.cols-1 || cy >= gray.rows-1)
                    break;

                float g = std::exp(-(d*d)/(2*sigma_m*sigma_m));
                lic.at<float>(y,x) += g * diff_norm.at<float>(int(cy), int(cx));
                w.at<float>(y,x)   += g;
            }
        }
    }

    cv::max(w, 1e-6f, w);
    lic /= w;



    out.create(diff_norm.size(), CV_32FC1);

    int rows = diff_norm.rows;
    int cols = diff_norm.cols * diff_norm.channels();
    float* ptr_in;
    float* ptr_out;

    for (int x = 0; x < rows; x++)
    {
        ptr_in = lic.ptr<float>(x);
        ptr_out = out.ptr<float>(x);

        for (int y = 0; y < cols; y++){
            if (ptr_in[y] >= thres){
                ptr_out[y] = 1.0f;
            } else {
                if (phi == -1.0f){
                    ptr_out[y] = 0;
                } else {
                    ptr_out[y] = 1.0f + std::tanh(phi * (ptr_in[y] - thres));
                }
            }
        }
    }


    cv::Mat aa(out.size(), CV_32F, 0.0f);
    cv::Mat wa(out.size(), CV_32F, 0.0f);

    int radiusA = int(2 * sigma_a);

    for (int y = 0; y < out.rows; ++y) {
        for (int x = 0; x < out.cols; ++x) {

            float cx = x;
            float cy = y;

            // center sample
            aa.at<float>(y,x) += out.at<float>(y,x);
            wa.at<float>(y,x)  += 1.0f;

            // forward
            for (int d = 1; d <= radiusA; ++d) {
                float txp = tx.at<float>(int(cy), int(cx));
                float typ = ty.at<float>(int(cy), int(cx));

                cx += txp;
                cy += typ;

                if (cx < 0 || cy < 0 || cx >= out.cols-1 || cy >= out.rows-1)
                    break;

                float g = std::exp(-(d*d)/(2*sigma_a*sigma_a));
                aa.at<float>(y,x) += g * out.at<float>(int(cy), int(cx));
                wa.at<float>(y,x)  += g;
            }

            // backward
            cx = x;
            cy = y;
            for (int d = 1; d <= radiusA; ++d) {
                float txp = tx.at<float>(int(cy), int(cx));
                float typ = ty.at<float>(int(cy), int(cx));

                cx -= txp;
                cy -= typ;

                if (cx < 0 || cy < 0 || cx >= out.cols-1 || cy >= out.rows-1)
                    break;

                float g = std::exp(-(d*d)/(2*sigma_a*sigma_a));
                aa.at<float>(y,x) += g * out.at<float>(int(cy), int(cx));
                wa.at<float>(y,x)  += g;
            }
        }
    }

    cv::max(wa, 1e-6f, wa);
    aa /= wa;

    cv::normalize(aa, aa, 0.0f, 1.0f, cv::NORM_MINMAX);
    cv::Mat mask;
    aa.convertTo(mask, CV_8UC1, 255.0);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);

    float alpha = static_cast<float>(params[8]);
    float beta  = 1.0f - alpha;

    cv::addWeighted(in, alpha, mask, beta, 0.0, out);
}


void build_funcs()
{
    g_func_map["dog_e"] = &dogExtended;
    g_function_data.push_back({
        "dog_e",
        "Extended DoG",
        {
           { "Blur", 1.4, 0, 10, 2, 0.1 },
           { "K", 1.5, 1, 10, 2, 0.1 },
           { "Threshold", 0.6, 0, 1, 2, 0.05 },
           { "Sharpen", 1, 0, 100, 2, 0.05 },
           { "Tone Smoother", 3, -1, 100, 3, 1 },
        }
    });

    g_func_map["dog_s"] = &dogSuper;
    g_function_data.push_back({
        "dog_s",
        "Super DoG",
        {
            { "Edge Blur", 1.4, 0, 10, 2, 0.1 },
            { "K", 1.5, 1, 10, 2, 0.1 },
            { "Threshold", 0.6, 0, 1, 2, 0.05 },
            { "Sharpen", 1, 0, 100, 2, 0.05 },
            { "Tone Smoother", 3, -1, 100, 3, 1 },
            { "Tensor Blur", 1.4, 0, 10, 2, 0.1 },
            { "LIC Blur", 1.4, 0, 10, 2, 0.1 },
            { "AA Blur", 1.4, 0, 10, 2, 0.1 },
            { "Blend Color", 0.5, 0, 1, 2, 0.1 }
        }
    });

    g_func_map["grayscale"] = &grayscale;
    g_function_data.push_back({
        "grayscale",
        "Grayscale",
        {}
    });

    g_func_map["sobel_g"] = &sobelGray;
    g_function_data.push_back({
        "sobel_g",
        "Sobel Gray",
        {
            { "Threshold", 100, -1, 255, 0, 1.0 },
            { "Threshold Mode", 0, 0, 2, 0, 1.0 }
        }
    });

    g_func_map["sobel_rgb"] = &sobelRGB;
    g_function_data.push_back({
        "sobel_rgb",
        "Sobel RGB",
        {
            { "Threshold", 100, -1, 255, 0, 1.0 },
            { "Threshold Mode", 0, 0, 2, 0, 1.0 }
        }
    });

    g_func_map["quantize"] = &quantize;
    g_function_data.push_back({
        "quantize",
        "Quantize",
        {
            { "Divisor", 100, 1, 255, 0, 1.0 }
        }
    });

    g_func_map["add"] = &add;
    g_function_data.push_back({
        "add",
        "Add",
        {
            { "Constant", 100, 1, 255, 0, 1.0 }
        }
    });

    g_func_map["sub"] = &sub;
    g_function_data.push_back({
        "sub",
        "Subtract",
        {
            { "Constant", 100, 1, 255, 0, 1.0 }
        }
    });

    g_func_map["divide"] = &divide;
    g_function_data.push_back({
        "divide",
        "Divide",
        {
            { "Constant", 2, 1, 255, 3, 1.0 }
        }
    });

    g_func_map["mul"] = &mul;
    g_function_data.push_back({
        "mul",
        "Multiply",
        {
            { "Constant", 2, 1, 255, 3, 1.0 }
        }
    });

    g_func_map["boxBlur"] = &boxBlur;
    g_function_data.push_back({
        "boxBlur",
        "Box Blur",
        {
            { "Size", 5, 3, 33, 0, 1.0 }
        }
    });

    g_func_map["gauBlur"] = &gauBlur;
    g_function_data.push_back({
        "gauBlur",
        "Gaussian Blur",
        {
            { "Sigma", 1, 1, 10, 2, 1.0 }
        }
    });
}
