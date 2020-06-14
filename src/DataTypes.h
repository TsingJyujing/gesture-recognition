#ifndef GESTURERECO_DATATYPES_H
#define GESTURERECO_DATATYPES_H

#include <vector>
#include <iostream>

using namespace std;


template<typename T>
class Point {
    T x;
    T y;
public:
    Point(T x, T y) : x(x), y(y) {}

    T getX() const {
        return x;
    }

    T getY() const {
        return y;
    }
};

template<typename T>
class Box {
    Point<T> leftTop;
    T w;
    T h;
public:
    Box(Point<T> leftTop, T w, T h);

    T getLeft() const;

    T getRight() const;

    T getTop() const;

    T getBottom() const;

    T getArea() const;

    T getW() const;

    T getH() const;

    float intersectOverUnion(const Box &b) const;

};

template<typename T>
class DetectionResult {
    Box<T> box;
    vector<Point<T>> keyPoints;
public:
    DetectionResult(Box<T> box, vector<Point<T>> keyPoints);

    const Box<T> &getBox() const;

    const vector<Point<T>> &getKeyPoints() const;
};

template<typename T>
class NMSCandidate {
    int index;
    float p;
    DetectionResult<T> detectionResult;
public:
    NMSCandidate(int index, float p, DetectionResult<T> detectionResult);

    float getP() const;

    const DetectionResult<T> &getDetectionResult() const;
};


// Implementations

template<typename T>
Box<T>::Box(Point<T> leftTop, T w, T h):leftTop(leftTop), w(w), h(h) {}

template<typename T>
T Box<T>::getW() const {
    return w;
}

template<typename T>
T Box<T>::getH() const {
    return h;
}

template<typename T>
T Box<T>::getLeft() const {
    return this->leftTop.getX();
}

template<typename T>
T Box<T>::getRight() const {
    return this->leftTop.getX() + this->w;
}

template<typename T>
T Box<T>::getTop() const {
    return this->leftTop.getY();
}

template<typename T>
T Box<T>::getBottom() const {
    return this->leftTop.getY() + this->h;
}

template<typename T>
T Box<T>::getArea() const {
    return this->getW() * this->getH();
}

template<typename T>
float Box<T>::intersectOverUnion(const Box &b) const {
    const int xIntersect = (int) (min(b.getRight(), this->getRight()) - max(b.getLeft(), this->getLeft()));
    const int yIntersect = (int) (min(b.getBottom(), this->getBottom()) - max(b.getTop(), this->getTop()));
    if (xIntersect <= 0 || yIntersect <= 0) {
        return 0;
    } else {
        const int intersectArea = xIntersect * yIntersect;
        return (float) intersectArea / (float) (b.getArea() + this->getArea() - intersectArea);
    }
}


template<typename T>
DetectionResult<T>::DetectionResult(Box<T> box, vector<Point<T>> keyPoints):box(box), keyPoints(keyPoints) {}

template<typename T>
const Box<T> &DetectionResult<T>::getBox() const {
    return box;
}

template<typename T>
const vector<Point<T>> &DetectionResult<T>::getKeyPoints() const {
    return keyPoints;
}

template<typename T>
NMSCandidate<T>::NMSCandidate(int index, float p, DetectionResult<T> detectionResult):
        index(index), p(p), detectionResult(detectionResult) {}

template<typename T>
float NMSCandidate<T>::getP() const {
    return p;
}

template<typename T>
const DetectionResult<T> &NMSCandidate<T>::getDetectionResult() const {
    return detectionResult;
}



#endif //GESTURERECO_DATATYPES_H
