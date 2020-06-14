//
// Created by Yifan Yuan on 2020/06/02.
//

#include "HandProcessor.h"


void split(const string &s, vector<string> &tokens, const string &delimiters = ",") {
    string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (string::npos != pos || string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));//use emplace_back after C++11
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}

PalmDetector::PalmDetector(
        const char *modelPath,
        const char *anchorsPath,
        unsigned int batchSize,
        int numThreads
) {
    this->model = FlatBufferModel::BuildFromFile(modelPath, nullptr);
    InterpreterBuilder builder(*model, this->resolver);
    builder(&(this->interpreter));
    if (interpreter == nullptr) {
        throw bad_alloc();
    }
    if (batchSize > 1) {
        fprintf(stdout, "Batch size set to %d\n", batchSize);
        const vector<int> resize_input = {(int) batchSize, 256, 256, 3};
        this->interpreter->ResizeInputTensor(
                interpreter->inputs()[0],
                resize_input
        );
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        throw bad_alloc();
    }

    this->inputTensor = interpreter->tensor(
            interpreter->inputs()[0]
    );
    this->outputRegressor = interpreter->tensor(
            interpreter->outputs()[0]
    );
    this->outputClassificator = interpreter->tensor(
            interpreter->outputs()[1]
    );
    fprintf(stdout, "Thread count set to %d\n", numThreads);
    this->interpreter->SetNumThreads(numThreads);

    this->anchors = vector<tuple<float, float>>();
    ifstream anchorsFile(anchorsPath);
    std::string line;
    while (std::getline(anchorsFile, line)) {
        vector<string> tokens = vector<string>();
        split(line, tokens);
        this->anchors.emplace_back(
                stof(tokens[0]) * 256,
                stof(tokens[1]) * 256
        );
    }
    size_t classificatorDim = this->outputClassificator->bytes / sizeof(float);
    size_t regressorDim = this->outputRegressor->bytes / (18 * sizeof(float));
    size_t anchorsDim = this->anchors.size();
    if (classificatorDim != regressorDim || classificatorDim != anchorsDim) {
        fprintf(
                stderr,
                "Dimension Details: classificatorDim=%ld regressorDim=%ld anchorsDim=%ld \n",
                classificatorDim,
                regressorDim,
                anchorsDim
        );
        throw HandProcessorException("Dimension not correct.");
    }
}


tflite::Interpreter &PalmDetector::getInterpreter() {
    return *(this->interpreter);
}


TfLiteTensor &PalmDetector::getInputTensor() {
    return *(this->inputTensor);
}

float sigmoid(float x) {
    // FIXME while x too large or too small, return 1/-1 directly
    return 1.0f / (exp(-x) + 1.0f);
}

vector<DetectionResult<float>> PalmDetector::getResult(float probThreshold, float iouThreshold) {
    const size_t N = anchors.size();
    auto candidates = list<NMSCandidate<float>>();
    for (size_t i = 0; i < N; i++) {
        float prob = sigmoid(getClassificatorValue(i));
        if (prob > probThreshold) {
            float ax = get<0>(anchors[i]);
            float ay = get<1>(anchors[i]);
            float cx = getRegressorValue(i, 0) + ax;
            float cy = getRegressorValue(i, 1) + ay;
            float w = getRegressorValue(i, 2);
            float h = getRegressorValue(i, 3);
            auto candidateBox = Box<float>(Point<float>(cx - 0.5 * w, cy - 0.5 * h), w, h);
            auto keyPoints = vector<Point<float>>();
            for (int k = 0; k < 7; k++) {
                keyPoints.emplace_back(
                        getRegressorValue(i, k * 2 + 4 + 0) + ax,
                        getRegressorValue(i, k * 2 + 4 + 1) + ay
                );
            }
            candidates.emplace_back(
                    i,
                    prob,
                    DetectionResult<float>(
                            candidateBox,
                            keyPoints
                    )
            );
        }
    }
    vector<DetectionResult<float>> result = vector<DetectionResult<float>>();
    result.reserve(candidates.size());

    candidates.sort([](const NMSCandidate<float> &a, const NMSCandidate<float> &b) -> bool {
        return a.getP() > b.getP();
    });

    while (!candidates.empty()) {
        const NMSCandidate<float> &base = candidates.front();
        candidates.pop_front();
        candidates.remove_if([base, iouThreshold](const NMSCandidate<float> &x) -> bool {
            return base.getDetectionResult().getBox().intersectOverUnion(
                    x.getDetectionResult().getBox()
            ) > iouThreshold;
        });
        result.push_back(base.getDetectionResult());
    }

    return result;
}

float PalmDetector::getClassificatorValue(int i) {
    return this->outputClassificator->data.f[i];
}

float PalmDetector::getRegressorValue(int i, int j) {
    return this->outputRegressor->data.f[i * 18 + j];
}

HandProcessorException::HandProcessorException(const char *reason) {
    this->reason = reason;
}

HandProcessorException::HandProcessorException() {
    this->reason = "";
}

DelayTrigger::DelayTrigger(float delaySeconds) {
    this->delayus = (unsigned long) (delaySeconds * 1e6);
}

bool DelayTrigger::syncTrigger(bool conditionSatisfied) {
    auto tick = (chrono::system_clock::now() - this->startTick).count();
    if (conditionSatisfied) {
        this->lastTick = tick;
        this->fire = true;
    } else {
        if ((tick - this->lastTick) > this->delayus && this->fire) {
            this->fire = false;
            return true;
        }
    }
    return false;
}


