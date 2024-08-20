#include "YOLOv5.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "utils.h"

using namespace yolov5;

int main(int argc, char **argv)
{

    std::string model_path = "/home/iot/snpe_inference_MobileNet_SSD/models/model_human/three_node_v6.2/snpe_2.16/yolov5s_640_kyle.dlc"; // Replace with your actual model path

    // Set up object detection configuration
    ObjectDetectionConfig config;
    config.model_path = model_path;
    config.runtime = CPU;                      // Change this according to your runtime
    config.labels = 6;                         // Number of classes in the model (includes bounding box information)
    config.grids = 6300;                       // Total number of grids (this may vary depending on your YOLOv5 model)
    config.inputLayers.emplace_back("images"); // Input layer name

    config.outputLayers.emplace_back("/model.24/Sigmoid"); // Output layer name
    config.outputLayers.emplace_back("/model.24/Sigmoid_1");
    config.outputLayers.emplace_back("/model.24/Sigmoid_2");

    config.outputTensors.emplace_back("output"); // Output tensor name
    config.outputTensors.emplace_back("332");
    config.outputTensors.emplace_back("334");

    // Initialize object detection
    ObjectDetection detector;
    if (!detector.Init(config))
    {
        std::cerr << "@@@@@@ Failed to initialize object detection..." << std::endl;
        return -1;
    }

    // Open webcam
    // 自動偵測可用攝像頭----------------------------------------------
    int camID = 0;
    for (int device_id = 0; device_id < 10; device_id++)
    {
        cv::VideoCapture cap(device_id);
        std::cout << "VideoCapture: " << device_id << std::endl;
        if (cap.isOpened())
        {
            double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);

            std::cout << "\nUSB Camera Parameters (Device " << device_id << "):" << std::endl;
            std::cout << "Resolution: " << width << "x" << height << std::endl;
            std::cout << "FPS: " << fps << std::endl;

            camID = device_id;
            cap.release();
            break;
        }
    }

    std::ostringstream buff;
    buff << camID;

    cv::VideoCapture cap;
    cap.open("/dev/video" + buff.str());
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, img_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, img_height);
    cap.set(cv::CAP_PROP_FPS, 60);
    double fps2 = cap.get(cv::CAP_PROP_FPS);
    std::cout << "\nfps : " << fps2 << std::endl;
    // -----------------------------------------------------------------------------

    cv::VideoWriter out("appsrc ! videoconvert ! "
                        "waylandsink fullscreen=true async=true sync=false",
                        cv::CAP_GSTREAMER, 0, 30, cv::Size(img_width, img_height), true);

    while (true)
    {
        long start_pre = GetTimeStamp_ms();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            std::cerr << "@@@@@@ Failed to capture frame..." << std::endl;
            break;
        }

        // Perform object detection
        std::vector<ObjectData> results;
        if (!detector.Detect(frame, results))
        {
            std::cerr << "@@@@@@ Failed to detect objects..." << std::endl;
            continue;
        }

        // Draw detected objects on the frame
        for (const auto &obj : results)
        {
            cv::rectangle(frame, obj.bbox, cv::Scalar(0, 255, 0), 2);

            std::string label = "Confidence: " + std::to_string(obj.confidence);
            cv::putText(frame, label, cv::Point(obj.bbox.x, obj.bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            std::string inferenceTimeText = "Inference Time: " + std::to_string(obj.time_cost) + " ms";
            cv::putText(frame, inferenceTimeText, cv::Point(obj.bbox.x, obj.bbox.y - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        }

        // Calculate FPS
        float diff_sum = GetTimeStamp_ms() - start_pre;
        float fps = 1 / (diff_sum / 1000);

        // Display FPS
        std::string fpsText = "FPS: " + std::to_string(fps);
        cv::putText(frame, fpsText, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        // Show the frame
        out.write(frame);

        // Exit on ESC key press
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    // Deinitialize object detection
    detector.Deinit();
    return 0;
}
