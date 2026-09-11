#include <vector>
#include <memory>
#include <string>
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




int main(){
    std::vector<std::shared_ptr<Sensor>> all_sensors = {
        std::make_shared<Sensor>(30.0), 
        std::make_shared<Sensor>(35.0), 
        std::make_shared<Sensor>(40.0), 
        std::make_shared<Sensor>(45.0), 
        std::make_shared<Sensor>(50.0)
    };

    std::vector<std::shared_ptr<Sensor>> triggered_sensors;

    double temp = 36.7;
    int i = 0;
    for(auto& s : all_sensors){
        if (s->isTriggered(temp)){
            triggered_sensors.push_back(s);
        }
        std::cout << "Sensor" << ++i << "_count : " << s.use_count() << std::endl;
    }

    std::cout << "-----------------------------------------" << std::endl;


    all_sensors.clear();

    std::cout << "-----------------------------------------" << std::endl;


    triggered_sensors.clear();


}