#include <map>
#include <vector>
#include <string>
#include <iostream>
#include "Sensor.h"

double average(double* arr, int size){
    double sum = 0;
    for(int i=0;i<size;i++)
        sum += arr[i];

    return sum/size;
}

double average(const std::vector<double>& arr){
    double sum = 0;
    for(double x : arr)
        sum += x;

    return sum/arr.size();
}

// double findMax(double* arr, int size){
//     double max_value = arr[0];

//     for(int i=1;i<size;i++)
//         if (arr[i] > max_value)
//             max_value = arr[i];

//     return max_value;
// }

double findMax(const std::vector<double>& arr){
    double max_value = arr[0];
    for(double x : arr)
        if (x > max_value)
            max_value = x;

    return max_value;
}

int main(){
    // double readings[10] = {-9, 15, 5, -1, 49, 39, 33, 45, 24, 16};
    std::vector<double> readings = {-9, 15, 5, -1, 49, 39, 33, 45, 24, 16};
    int arr_size = std::size(readings);

    // double avg = average(readings, arr_size);
    // double max_value = findMax(readings, arr_size);
    double avg = average(readings);
    double max_value = findMax(readings);

    Sensor s(30.0);

    int trigger_count = 0;
    for(int i=0;i<arr_size;i++)
        trigger_count += s.isTriggered(readings[i]);
    
    std::cout << "avg : " << avg << ", max : " << max_value << ", count : " << trigger_count << std::endl;
}