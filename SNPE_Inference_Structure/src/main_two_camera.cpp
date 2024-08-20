#include "YOLOv5.h"
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include "utils.h"

using namespace yolov5;

float img_width = 1280;
float img_height = 720;

// Structure to hold capture arguments
struct CaptureArgs
{
    cv::VideoCapture *cap;
    cv::Mat *frame;
    pthread_mutex_t *frameMutex;
    ObjectDetection *detector; // Pointer to the detector for inference
    const char *name;
};

pthread_mutex_t frameMutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t frameMutex2 = PTHREAD_MUTEX_INITIALIZER;
cv::Mat frame1, frame2;
bool running = true;

// Thread function to capture frames and apply object detection
void *captureFrame(void *args)
{
    CaptureArgs *captureArgs = (CaptureArgs *)args;
    cv::VideoCapture *cap = captureArgs->cap;
    cv::Mat *frame = captureArgs->frame;
    pthread_mutex_t *frameMutex = captureArgs->frameMutex;
    ObjectDetection *detector = captureArgs->detector;
    const char *name = captureArgs->name;

    while (running)
    {
        long start_pre = GetTimeStamp_ms();

        cv::Mat tempFrame;
        *cap >> tempFrame;
        if (tempFrame.empty())
        {
            printf("Failed to capture frame from %s\n", name);
            continue;
        }

        // Perform object detection
        std::vector<ObjectData> results;
        if (!detector->Detect(tempFrame, results))
        {
            std::cerr << "@@@@@@ Failed to detect objects from " << name << std::endl;
            continue;
        }

        // Draw detected objects on the frame
        for (const auto &obj : results)
        {
            cv::rectangle(tempFrame, obj.bbox, cv::Scalar(0, 255, 0), 2);

            std::string label = "Confidence: " + std::to_string(obj.confidence);
            cv::putText(tempFrame, label, cv::Point(obj.bbox.x, obj.bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            std::string inferenceTimeText = "Inference Time: " + std::to_string(obj.time_cost) + " ms";
            cv::putText(tempFrame, inferenceTimeText, cv::Point(obj.bbox.x, obj.bbox.y - 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
        }

        // Calculate FPS
        float diff_sum = GetTimeStamp_ms() - start_pre;
        float fps = 1 / (diff_sum / 1000);
        std::cout << "FPS DSP: " << fps << std::endl;

        pthread_mutex_lock(frameMutex);
        *frame = tempFrame.clone();
        pthread_mutex_unlock(frameMutex);

        // printf("Frame processed from %s\n", name);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    std::string model_path = "/data/local/model_path/three_node_v6.2/snpe_2.16/kyle/yolov5s_640_kyle_quantized.dlc"; // Replace with your actual model path

    // Set up object detection configuration
    ObjectDetectionConfig config;
    config.model_path = model_path;
    config.runtime = DSP;                                  // Change this according to your runtime
    config.labels = 6;                                     // Number of classes in the model (includes bounding box information)
    config.grids = 25200;                                  // Total number of grids (this may vary depending on your YOLOv5 model)
    config.inputLayers.emplace_back("images");             // Input layer name
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

    // Set ScoreThreshold
    float person_confidence = 0.65;
    float nms = 0.50;
    detector.SetScoreThreshold(person_confidence, nms);

    int camID1 = -1, camID2 = -1;

    // Detect available cameras
    for (int device_id = 0; device_id < 10; ++device_id)
    {
        cv::VideoCapture cap(device_id);
        if (cap.isOpened())
        {
            double width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            double height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);

            std::cout << "\nUSB Camera Parameters (Device " << device_id << "):" << std::endl;
            std::cout << "Resolution: " << width << "x" << height << std::endl;
            std::cout << "FPS: " << fps << std::endl;

            if (camID1 == -1)
            {
                camID1 = device_id;
            }
            else if (camID2 == -1)
            {
                camID2 = device_id;
                cap.release();
                break;
            }
            cap.release();
        }
    }

    if (camID1 == -1 || camID2 == -1)
    {
        std::cerr << "Error: Less than two cameras detected" << std::endl;
        return -1;
    }

    std::ostringstream buff1, buff2;
    buff1 << camID1;
    buff2 << camID2;

    cv::VideoCapture cap1("/dev/video" + buff1.str());
    cv::VideoCapture cap2("/dev/video" + buff2.str());

    if (!cap1.isOpened() || !cap2.isOpened())
    {
        std::cerr << "Error: Could not open one or both cameras" << std::endl;
        return -1;
    }

    // Set camera properties
    cap1.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap1.set(cv::CAP_PROP_FRAME_WIDTH, img_width);
    cap1.set(cv::CAP_PROP_FRAME_HEIGHT, img_height);
    cap1.set(cv::CAP_PROP_FPS, 30);
    double fps1 = cap1.get(cv::CAP_PROP_FPS);
    std::cout << "\nInit Camera 1 FPS : " << fps1 << std::endl;

    cap2.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap2.set(cv::CAP_PROP_FRAME_WIDTH, img_width);
    cap2.set(cv::CAP_PROP_FRAME_HEIGHT, img_height);
    cap2.set(cv::CAP_PROP_FPS, 30);
    double fps2 = cap2.get(cv::CAP_PROP_FPS);
    std::cout << "\nInit Camera 2 FPS : " << fps2 << std::endl;

    // Create VideoWriter object for GStreamer
    cv::VideoWriter out("appsrc ! videoconvert ! waylandsink fullscreen=false async=true sync=false",
                        cv::CAP_GSTREAMER, 0, 30, cv::Size(img_width * 2, img_height), true);

    if (!out.isOpened())
    {
        std::cerr << "Error: Could not open VideoWriter" << std::endl;
        return -1;
    }

    // Set up capture arguments
    CaptureArgs args1 = {&cap1, &frame1, &frameMutex1, &detector, "Camera 1"};
    CaptureArgs args2 = {&cap2, &frame2, &frameMutex2, &detector, "Camera 2"};

    // Create threads to capture frames and apply object detection
    pthread_t captureThread1, captureThread2;
    pthread_create(&captureThread1, NULL, captureFrame, (void *)&args1);
    pthread_create(&captureThread2, NULL, captureFrame, (void *)&args2);

    while (true)
    {
        cv::Mat displayFrame1, displayFrame2, combinedFrame;

        // Lock the frame mutex and get the frames
        pthread_mutex_lock(&frameMutex1);
        displayFrame1 = frame1.clone();
        pthread_mutex_unlock(&frameMutex1);

        pthread_mutex_lock(&frameMutex2);
        displayFrame2 = frame2.clone();
        pthread_mutex_unlock(&frameMutex2);

        //---------------------------------------------------------combinedFrame
        // Ensure frames are not empty
        if (!displayFrame1.empty() && !displayFrame2.empty())
        {
            // Create a combined frame with the width of both frames combined
            combinedFrame.create(img_height, img_width * 2, displayFrame1.type());

            // Copy the frames side by side into the combined frame
            displayFrame1.copyTo(combinedFrame(cv::Rect(0, 0, img_width, img_height)));
            displayFrame2.copyTo(combinedFrame(cv::Rect(img_width, 0, img_width, img_height)));
        }
        //----------------------------------------------------------------------

        // Display the combined frame using GStreamer pipeline
        if (!combinedFrame.empty())
        {
            out.write(combinedFrame);
        }

        // Wait for 'q' key press for 30ms. If 'q' key is pressed, break the loop
        if (cv::waitKey(30) == 'q')
        {
            running = false;
            break;
        }
    }

    // Wait for threads to finish
    pthread_join(captureThread1, NULL);
    pthread_join(captureThread2, NULL);

    // Release the camera resources
    cap1.release();
    cap2.release();
    out.release();
    detector.Deinit();
    cv::destroyAllWindows();

    return 0;
}
