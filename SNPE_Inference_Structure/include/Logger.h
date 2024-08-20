#pragma once

#include <iostream>
#include <cstring>
#include <sstream>
#include <time.h>
#include <chrono>
#include <memory>

// #include "spdlog/spdlog.h"
// #include "spdlog/async.h"
// #include "spdlog/sinks/basic_file_sink.h"
// #include "spdlog/sinks/rotating_file_sink.h"
// #include "spdlog/sinks/stdout_color_sinks.h"

static inline int NowDateToInt()
{
    time_t now;
    time(&now);

    tm p;
    localtime_r(&now, &p);
    int now_date = (1900 + p.tm_year) * 10000 + (p.tm_mon + 1) * 100 + p.tm_mday;
    return now_date;
}

static inline int NowTimeToInt()
{
    time_t now;
    time(&now);

    tm p;
    localtime_r(&now, &p);

    int now_int = p.tm_hour * 10000 + p.tm_min * 100 + p.tm_sec;
    return now_int;
}

#define LOG_TRACE(fmt, ...) (std::printf(fmt, ##__VA_ARGS__), std::printf("\n"))
#define LOG_DEBUG(fmt, ...) (std::printf(fmt, ##__VA_ARGS__), std::printf("\n"))
#define LOG_INFO(fmt, ...) (std::printf(fmt, ##__VA_ARGS__), std::printf("\n"))
#define LOG_WARN(fmt, ...) (std::printf(fmt, ##__VA_ARGS__), std::printf("\n"))
#define LOG_ERROR(fmt, ...) (std::printf(fmt, ##__VA_ARGS__), std::printf("\n"))