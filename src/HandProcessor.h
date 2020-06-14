//
// Created by Yifan Yuan on 2020/06/02.
//

#ifndef GESTURERECO_HANDPROCESSOR_H
#define GESTURERECO_HANDPROCESSOR_H


#include <iostream>
#include <fstream>
#include <cmath>
#include <functional>
#include <list>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "DataTypes.h"

using namespace std;
using namespace tflite;


class HandProcessorException : public exception {
    const char *reason;
public:
    HandProcessorException();

    HandProcessorException(const char *reason);

    const char *what() const noexcept override {
        return this->reason;
    }
};


class PalmDetector {
    std::unique_ptr<tflite::Interpreter> interpreter;
    ops::builtin::BuiltinOpResolver resolver;
    unique_ptr<FlatBufferModel> model;
    vector<tuple<float, float>> anchors;

    TfLiteTensor *inputTensor;
    TfLiteTensor *outputRegressor;
    TfLiteTensor *outputClassificator;

    float getClassificatorValue(int i);

    float getRegressorValue(int i, int j);

public:
    /**
     * Initialize class
     * @param modelPath Path of BlazePalm model's tflite file
     * @param anchorsPath Path of anchors in the model
     * @param batchSize Inference batch size
     * @param numThreads TF Lite num thread
     */
    PalmDetector(
            const char *modelPath,
            const char *anchorsPath,
            unsigned int batchSize = 1,
            int numThreads = 4
    );

    vector<DetectionResult<float>> getResult(float probThreshold, float iouThreshold);

    Interpreter &getInterpreter();

    TfLiteTensor &getInputTensor();
};


class DelayTrigger {
    unsigned long delayus;
    unsigned long lastTick = -1;
    bool fire = false;
    chrono::system_clock::time_point startTick = chrono::system_clock::now();
public:
    DelayTrigger(float delaySeconds);

    bool syncTrigger(bool conditionSatisfied);
};

#endif //GESTURERECO_HANDPROCESSOR_H
