#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "Sensor.h"

void alarm_counter(const std::vector<double>& readings, const std::vector<Sensor>& sensors, std::map<int, int>& counter){
    for(size_t i=0;i<sensors.size();i++)
        for(double temp : readings)
            if (sensors[i].isTriggered(temp))
                counter[i]++;

}

void find_max(const std::map<int, int>& counter, int& max_value, int& max_index){
    for(const auto& pair : counter)
        if (pair.second > max_value){
            max_index = pair.first;
            max_value = pair.second; 
        }

}

int main(){

    std::vector<double> readings = {-9, 15, 5, -1, 49, 39, 33, 45, 24, 16};
    std::vector<Sensor> sensors = {Sensor(20.0), Sensor(30.0), Sensor(40.0)};
    std::map<int, int> counter;

    alarm_counter(readings, sensors, counter);

    int max_value = 0;
    int max_index = 0;
    
    find_max(counter, max_value, max_index);

    std::cout << "sensor" << max_index << " has max value " << max_value << std::endl;

}