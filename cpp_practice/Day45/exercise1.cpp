#include <string>
#include <memory>
#include <iostream>

class Sensor{
    public:
        explicit Sensor(double threshold);
        ~Sensor();
        bool isTriggered(double value) const;
    private:
        double threshold_;
};

Sensor::Sensor(double threshold) {
    threshold_ = threshold;
    std::cout << "Sensor 建構, threshold = " << threshold_ << std::endl;
}

Sensor::~Sensor(){
    std::cout << "Sensor 解構, threshold = " << threshold_ << std::endl;
}

bool Sensor::isTriggered(double value) const {
    return value > threshold_;
}

Sensor* createSensor_wrong(){
    Sensor s(20.0);
    return &s;
}

std::unique_ptr<Sensor> createSensor_correct(){
    std::unique_ptr<Sensor> p = std::make_unique<Sensor>(20.0);
    return p;
}

int main(){
    {
        std::unique_ptr<Sensor> p = std::make_unique<Sensor> (20.0);
    }
    std::cout << "scope out" << std::endl;

    {
        Sensor* p = new Sensor(20.0);
    }
    std::cout << "scope out" << std::endl;

    // Sensor* p = createSensor_wrong();
    // std::cout << "triggered? " << (*p).isTriggered(25.0) << std::endl;  // 存取已經死掉的記憶體

    std::unique_ptr<Sensor> p = createSensor_correct();
    std::cout << "triggered = " << p->isTriggered(25.0) << std::endl;  // 存取已經死掉的記憶體


}