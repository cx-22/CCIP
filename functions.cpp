#include "functions.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/opencv.hpp>
//#include "lime.hpp"

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>


std::unordered_map<std::string, FunctionPtr> g_func_map;
std::vector<FunctionData> g_function_data;


static const int bayer2[2 * 2] = {
    0, 2,
    3, 1
};

static const int bayer4[4 * 4] = {
    0, 8, 2, 10,
    12, 4, 14, 6,
    3, 11, 1, 9,
    15, 7, 13, 5
};

static const int bayer8[8 * 8] = {
    0, 32, 8, 40, 2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
    3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21
};

float GetBayer2(int x, int y) {
    return float(bayer2[(x % 2) + (y % 2) * 2]) * (1.0f / 4.0f) - 0.5f;
}

float GetBayer4(int x, int y) {
    return float(bayer4[(x % 4) + (y % 4) * 4]) * (1.0f / 16.0f) - 0.5f;
}

float GetBayer8(int x, int y) {
    return float(bayer8[(x % 8) + (y % 8) * 8]) * (1.0f / 64.0f) - 0.5f;
}


void grayscale(cv::Mat& in, cv::Mat& out, std::vector<float>&)
{

    int rows = in.rows;
    int cols = in.cols;
    uchar* ptr_in;
    uchar* ptr_out;
    uchar val;
    for (int x = 0; x < rows; x++)
    {
        ptr_in = in.ptr<uchar>(x);
        ptr_out = out.ptr<uchar>(x);

        for (int y = 0; y < cols; y++){

            int idx = y * 3;

            val = static_cast<uchar>((0.114f * ptr_in[idx]) + (0.587f * ptr_in[idx + 1]) + (0.299f * ptr_in[idx + 2]));
            ptr_out[idx] = val;
            ptr_out[idx + 1] = val;
            ptr_out[idx + 2] = val;
        }
    }
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


    float sigma_e = params[0];
    double k = static_cast<double>(params[1]);
    float thres = static_cast<float>(params[2]);
    int vec_size = static_cast<int>(params[3]);
    float tau = params[4];
    float phi = params[5];
    //float a2g = params[6];
    //int lic_step = params[7];

    /*
    cv::Mat mid, empty;
    lime::DoGParams dog_params = lime::DoGParams(k, sigma_e, tau, phi, lime::NPR_EDGE_FDOG);

    cv::Mat gray, gray_f, out_p;
    cv::cvtColor(in, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray_f, CV_32F);

    auto start = std::chrono::high_resolution_clock::now();

    lime::edgeFDoG(gray_f, out_p, empty, dog_params, vec_size, a2g, thres, lic_step);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cerr << "Function execution time: " << duration.count() << " seconds" << std::endl;


    cv::normalize(out_p, out_p, 0.0f, 1.0f, cv::NORM_MINMAX);

    std::cerr << "Center value: " << out_p.at<float>(out_p.rows/2, out_p.cols/2) << std::endl;


    cv::Mat temp1, temp2;
    out_p.convertTo(temp1, CV_8U, 255.0);
    */


    //*
    float sigma_c = params[6];
    float sigma_m = params[7];
    float sigma_a = params[8];


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
    //*/

    cv::normalize(aa, aa, 0.0f, 1.0f, cv::NORM_MINMAX);
    cv::Mat mask;
    aa.convertTo(mask, CV_8UC1, 255.0);
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);


    float alpha = static_cast<float>(params[9]);

    if (alpha > 0.0f){
        float sigma_b = static_cast<float>(params[10]);
        float beta  = 1.0f - alpha;

        cv::Mat color;

        if (sigma_b != -1){
            cv::GaussianBlur(in, color, cv::Size(0, 0), sigma_b);
        } else {
            color = in.clone();
        }

        cv::addWeighted(color, alpha, mask, beta, 0.0, out);
    } else {
        out = mask.clone();
    }


}


void paperTex(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    float beta = static_cast<float>(params[0]);
    float alpha  = 1.0f - beta;

    cv::Mat paper_source = cv::imread("assets/textures/paper.png");

    cv::Mat ratioed;
    cv::Size size(paper_source.cols, paper_source.rows);

    if (in.cols > paper_source.cols){
        size.width = in.cols;
    }
    if (in.rows > paper_source.rows){
        size.height = in.rows;
    }

    cv::resize(paper_source, ratioed, size, 0, 0, cv::INTER_LINEAR);
    cv::Rect roi = cv::Rect(0, 0, in.cols, in.rows);
    cv::Mat paper_cropped = ratioed(roi);

    cv::addWeighted(in, alpha, paper_cropped, beta, 0.0, out);
}

void sharpen(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    float val = static_cast<float>(params[0]);

    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) << 0, 0, 0,
                                                0, 1, 0,
                                                0, 0, 0);

    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) << 0, 1, 0,
                                                1, 1, 1,
                                                0, 1, 0);

    cv::Mat kernel_z = kernel_x + ((kernel_x - (kernel_y / val)) * val);

    cv::filter2D(in, out, -1, kernel_z, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
}

void saturate(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    cv::Mat hsv_image;
    cv::cvtColor(in, hsv_image, cv::COLOR_BGR2HSV);

    float saturation_scale = static_cast<float>(params[0]);

    for (int i = 0; i < hsv_image.rows; ++i) {
        for (int j = 0; j < hsv_image.cols; ++j) {
            float sat = hsv_image.at<cv::Vec3b>(i, j)[1];
            sat *= saturation_scale;
            hsv_image.at<cv::Vec3b>(i, j)[1] = cv::saturate_cast<uchar>(sat);
        }
    }

    cv::cvtColor(hsv_image, out, cv::COLOR_HSV2BGR);
}

float clamp01(float v) {
    return std::min(1.0f, std::max(0.0f, v));
}

void bayerDither2(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    float strength = params[0];

    cv::Mat f32, norm, out_prime;
    cv::cvtColor(in, in, cv::COLOR_BGR2GRAY);
    in.convertTo(f32, CV_32F, 255.0);
    cv::normalize(f32, norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    out_prime = norm.clone();

    for (int y = 0; y < in.rows; ++y) {
        const float* src = norm.ptr<float>(y);
        float* dst = out_prime.ptr<float>(y);

        for (int x = 0; x < in.cols; ++x) {
            float b = GetBayer2(x, y);
            float v = src[x] + strength * b;
            dst[x] = clamp01(v);
        }
    }

    out_prime.convertTo(out, CV_8UC3, 255.0);
}


void bayerDither4(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    float strength = params[0];

    cv::Mat f32, norm, out_prime;
    cv::cvtColor(in, in, cv::COLOR_BGR2GRAY);
    in.convertTo(f32, CV_32F, 255.0);
    cv::normalize(f32, norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    out_prime = norm.clone();

    for (int y = 0; y < in.rows; ++y) {
        const float* src = norm.ptr<float>(y);
        float* dst = out_prime.ptr<float>(y);

        for (int x = 0; x < in.cols; ++x) {
            float b = GetBayer4(x, y);
            float v = src[x] + strength * b;
            dst[x] = clamp01(v);
        }
    }

    out_prime.convertTo(out, CV_8UC3, 255.0);
}

void bayerDither8(cv::Mat& in, cv::Mat& out, std::vector<float>& params){
    float strength = params[0];

    cv::Mat f32, norm, out_prime;
    cv::cvtColor(in, in, cv::COLOR_BGR2GRAY);
    in.convertTo(f32, CV_32F, 255.0);
    cv::normalize(f32, norm, 0.0f, 1.0f, cv::NORM_MINMAX);

    out_prime = norm.clone();

    for (int y = 0; y < in.rows; ++y) {
        const float* src = norm.ptr<float>(y);
        float* dst = out_prime.ptr<float>(y);

        for (int x = 0; x < in.cols; ++x) {
            float b = GetBayer8(x, y);
            float v = src[x] + strength * b;
            dst[x] = clamp01(v);
        }
    }

    out_prime.convertTo(out, CV_8UC3, 255.0);
}

void build_funcs()
{
    g_func_map["bd2"] = &bayerDither2;
    g_function_data.push_back({
        "bd2",
        "Bayer Dither 2",
        {
            { "Strength", 1.5, .01, 10, 2, 0.5 }
        }
    });

    g_func_map["bd8"] = &bayerDither8;
    g_function_data.push_back({
        "bd8",
        "Bayer Dither 8",
        {
            { "Strength", 1.5, .01, 10, 2, 0.5 }
        }
    });

    g_func_map["bd4"] = &bayerDither4;
    g_function_data.push_back({
        "bd4",
        "Bayer Dither 4",
        {
            { "Strength", 1.5, .01, 10, 2, 0.5 }
        }
    });

    g_func_map["saturate"] = &saturate;
    g_function_data.push_back({
        "saturate",
        "Saturate",
        {
            { "Scale", 1.5, 1, 10, 2, 0.5 }
        }
    });

    g_func_map["sharpen"] = &sharpen;
    g_function_data.push_back({
        "sharpen",
        "Sharpen",
        {
            { "Factor", 5, 5, 30, 2, 1 }
        }
    });

    g_func_map["paper_tex"] = &paperTex;
    g_function_data.push_back({
        "paper_tex",
        "Paper Texture",
        {
            { "Blend Color", 0.5, 0, 1, 2, 0.1 }
        }
    });

    g_func_map["dog_e"] = &dogExtended;
    g_function_data.push_back({
        "dog_e",
        "Extended DoG",
        {
           { "Blur", 1.4, 0, 10, 2, 0.1 },
           { "K", 1.5, 1, 10, 2, 0.1 },
           { "Threshold", 0.6, 0, 1, 2, 0.05 },
           { "Sharpen", 1, 0, 100, 2, 0.05 },
           { "Tone Smoother", 3, -1, 100, 3, 1 }
        }
    });

    g_func_map["dog_s"] = &dogSuper;
    g_function_data.push_back({
        "dog_s",
        "Super DoG",
        {
            { "Edge Blur", 0.4, 0, 30, 2, 0.1 },
            { "K", 1.5, 1, 10, 2, 0.1 },
            { "Threshold", 0.6, 0, 1, 2, 0.05 },
            { "Vec Size", 3, 3, 100, 1, 2 },
            { "Sharpen", 1, 0, 100, 2, 0.05 },
            { "Tone Smoother", 1, -1, 100, 3, 1 },
            //{ "A 2 G", 1, 0, 100, 2, 0.05 },
            //{ "LIC Step", 3, 1, 100, 3, 1 },
            //*
            { "Tensor Blur", 1, 0, 30, 2, 0.1 },
            { "LIC Blur", 3, 0, 30, 2, 0.1 }, //sigm
            { "AA Blur", 1, 0, 30, 2, 0.1 }, //sig a
            //*/
            { "Blend Color", 0.0, 0, 1, 2, 0.1 },
            { "Blend Blur", 5, -1, 30, 2, 0.1 }
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
