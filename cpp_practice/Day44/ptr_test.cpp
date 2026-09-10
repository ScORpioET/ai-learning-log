#include <iostream>
#include <string>

bool safeDivide(int a, int b, int* result){
    if (b == 0 || result == nullptr)
        return false;
    
    *result = a / b;

    return true;
}

int main(){
    int result = 0;
    
    if (safeDivide(6, 3, &result))
        std::cout << "boolean = " << "true" << ", result = " << result << std::endl;
    else
        std::cout << "boolean = " << "false" << "division by zero, skip or result is nullptr." << std::endl;

    if (safeDivide(6, 0, &result))
        std::cout << "boolean = " << "true" << ", result = " << result << std::endl;
    else
        std::cout << "boolean = " << "false" << ", error message:division by zero, skip or result is nullptr." << std::endl;
}