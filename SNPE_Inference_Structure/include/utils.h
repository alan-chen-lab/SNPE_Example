#pragma once

#include <algorithm>
#include <functional>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "Logger.h"

#if defined(_MSC_VER)
#ifdef DLL_EXPORTS
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __declspec(dllimport)
#endif
#elif __GNUC__ >= 4
#define EXPORT_API __attribute__((visibility("default")))
#else
#define EXPORT_API
#endif

// Inference hardware runtime.
typedef enum runtime
{
    CPU = 0,
    GPU,
    DSP,
    DSP_FIXED8,
    AIP
} runtime_t;

static float calcIoU(const cv::Rect &a, const cv::Rect &b)
{
    float xOverlap = std::max(
        0.,
        std::min(a.x + a.width, b.x + b.width) - std::max(a.x, b.x) + 1.);
    float yOverlap = std::max(
        0.,
        std::min(a.y + a.height, b.y + b.height) - std::max(a.y, b.y) + 1.);
    float intersection = xOverlap * yOverlap;
    float unio =
        (a.width + 1.) * (a.height + 1.) +
        (b.width + 1.) * (b.height + 1.) - intersection;
    return intersection / unio;
}

static int64_t GetTimeStamp_ms()
{
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp =
        std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now());
    std::time_t timestamp = tp.time_since_epoch().count();
    return timestamp;
}

static void split(const std::string &str,
                  std::vector<std::string> &tokens,
                  const char delim = ' ')
{
    tokens.clear();
    std::istringstream iss(str);
    std::string tmp;
    while (std::getline(iss, tmp, delim))
    {
        if (tmp != "")
        {
            tokens.emplace_back(std::move(tmp));
        }
    }
}

static std::string float2str(float number)
{
    std::ostringstream buff;
    buff << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    buff << number;
    return buff.str();
}

// 計算兩點距離
static double calculateDistance(std::pair<int, int> &x, std::pair<int, int> &y)
{
    return std::sqrt(std::pow(x.first - y.first, 2) +
                     std::pow(x.second - y.second, 2));
}
