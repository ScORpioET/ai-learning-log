#include "Sensor.h"


Sensor::Sensor(double threshold) {
    threshold_ = threshold;
}

bool Sensor::isTriggered(double value) const {
    return value > threshold_;
}


