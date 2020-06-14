#include <memory>
#include <chrono>

#include <cstdio>
#include <execinfo.h>
#include <csignal>
#include <cstdlib>
#include <unistd.h>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "HandProcessor.h"
#include "SystemEvent.h"
#include "DataTypes.h"


using namespace cv;
using namespace std;

Mat GetSquareImage(const Mat &img, int target_width = 256) {
    int width = img.cols, height = img.rows;
    Mat square = Mat::zeros(target_width, target_width, img.type());
    int max_dim = (width >= height) ? width : height;
    float scale = ((float) target_width) / max_dim;
    Rect roi;

    if (width >= height) {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = (target_width - roi.height) / 2;
    } else {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = (target_width - roi.width) / 2;
    }

    resize(img, square(roi), roi.size());

    return square;
}

#pragma ide diagnostic ignored "EndlessLoop"

/**
 * Show video steam from camera
 * @param video_device_id camera ID
 */
void videoLoop(int video_device_id) {
    PalmDetector hp(
            "model/BlazePalm.tflite",
            "model/BlazePalmAnchors.csv"
    );
    VideoCapture capture(video_device_id);
    capture.set(CAP_PROP_EXPOSURE, 4);
    const auto start_tick = chrono::system_clock::now();
    long int count = 0;
    list<cv::Point> points = list<cv::Point>();
    auto delayTrigger = DelayTrigger(0.3);

    while (true) {
        Mat frame;
        capture >> frame;
        Mat resizedImage = GetSquareImage(frame);
        size_t array_size = resizedImage.rows * resizedImage.cols * resizedImage.channels();
        if (points.size() > 20) {
            points.pop_front();
        }
        if (hp.getInputTensor().bytes == (sizeof(float) * array_size)) {
            float *model_buffer = hp.getInputTensor().data.f;
            for (int i = 0; i < array_size; i++) {
                model_buffer[i] = (((float) resizedImage.data[i]) - 127.5f) / 127.5f;
            }
            hp.getInterpreter().Invoke();
            try {
                auto result = hp.getResult(0.99, 0.3);
                float maxArea = 0;
                if (!result.empty()) {
                    for (auto &it : result) {
                        auto area = it.getBox().getArea();
                        auto scalar = area > 5000 ? Scalar(255, 0, 255) : Scalar(0, 255, 255);
                        rectangle(
                                resizedImage,
                                cv::Point(it.getBox().getLeft(), it.getBox().getTop()),
                                cv::Point(it.getBox().getRight(), it.getBox().getBottom()),
                                scalar
                        );
                        points.emplace_back(
                                (it.getBox().getLeft() + it.getBox().getRight()) * 0.5,
                                (it.getBox().getTop() + it.getBox().getBottom()) * 0.5
                        );

                        if (area > maxArea) {
                            maxArea = area;
                        }
                    }
                }
                for (auto &it:points) {
                    circle(resizedImage, it, 2, Scalar(0, 255, 255));
                }
                if (delayTrigger.syncTrigger(maxArea > 5000)) {
                    cout << "Firing space key." << endl;
                    MacSystemCall::pressSingleKey(kVK_Space);
                }
            } catch (length_error &lex) {
                fprintf(stderr, "Error while decoding result caused by allocate error.");
            }
        } else {
            fprintf(stderr,
                    "Model buffer: %ld bytes not equals to image buffer %ld bytes\n",
                    hp.getInputTensor().bytes,
                    sizeof(float) * array_size
            );
        }
        count++;

        if (count % 20 == 1) {
            auto duration = chrono::system_clock::now() - start_tick;
            auto used_ms = duration.count() / 1000;
            fprintf(
                    stdout,
                    "Used %lld ms to run %ld inferences, %lld ms/frame\n",
                    used_ms,
                    count,
                    used_ms / count
            );

        }
        cv::Mat imageFlipped;
        cv::flip(resizedImage, imageFlipped, 1);
        imshow("Reading Video Stream", imageFlipped);
        waitKey(1);    //Delay 30
    }
}

void handler(int sig) {
    void *array[10];
    size_t size;
    // get void*'s for all entries on the stack
    size = backtrace(array, 10);
    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

int main(int argc, char *argv[]) {
    signal(SIGSEGV, handler);
    try {
        videoLoop(0);
    } catch (exception &ex) {
        cerr << "Video Loop execution failed:" << ex.what() << endl;
    }
    return 0;
}

