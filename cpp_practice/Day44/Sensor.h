#pragma once
class Sensor{
    public:
        explicit Sensor(double threshold);
        bool isTriggered(double value) const;
    private:
        double threshold_;
};