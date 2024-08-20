#include "YOLOv5.h"
// #include "MSRCR.h"
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
    config.grids = 25200;                      // Total number of grids (this may vary depending on your YOLOv5 model)
    config.inputLayers.emplace_back("images"); // Input layer name

    config.outputLayers.emplace_back("/model.24/Sigmoid"); // Output layer name
    config.outputLayers.emplace_back("/model.24/Sigmoid_1");
    config.outputLayers.emplace_back("/model.24/Sigmoid_2");

    config.outputTensors.emplace_back("output"); // Output tensor name
    config.outputTensors.emplace_back("332");
    config.outputTensors.emplace_back("334");

    // ---------------------------------------------------v1
    // // Initialize object detection
    // ObjectDetection detector;
    // if (!detector.Init(config))
    // {
    //     std::cerr << "@@@@@@ Failed to initialize object detection..." << std::endl;
    //     return -1;
    // }

    // // Set ScoreThreshold
    // float person_confidence = 0.65;
    // float nms = 0.55;
    // detector.SetScoreThreshold(person_confidence, nms);
    // ---------------------------------------------------v1

    //----------------------------------------------------v2
    // Initialize object detection && Set ScoreThreshold
    float person_confidence = 0.50;
    float nms = 0.50;
    std::vector<std::shared_ptr<yolov5::ObjectDetection>> detect_person;
    for (int i = 0; i < 1; i++)
    {
        std::shared_ptr<yolov5::ObjectDetection> detector = std::shared_ptr<yolov5::ObjectDetection>(new yolov5::ObjectDetection());
        detector->Init(config);
        detector->SetScoreThreshold(person_confidence, nms);
        detect_person.push_back(detector);
    }

    //----------------------------------------------------v2

    // Open webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "@@@@@@ Failed to open webcam..." << std::endl;
        return -1;
    }
    cv::namedWindow("YOLOv5 Object Detection", cv::WINDOW_AUTOSIZE);

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
        if (!detect_person[0]->Detect(frame, results))
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
        cv::imshow("YOLOv5 Object Detection", frame);

        // Exit on ESC key press
        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }

    // Deinitialize object detection
    detect_person[0]->Deinit();
    return 0;
}
