#include <iostream>
#include <string>
#include <map>

std::string classify(double temp){
    if (temp < 10)
        return "cold";
    else if(10 <= temp && temp <= 30)
        return "normal";
    else
        return "hot";
}

int main(){

    std::map<std::string, int> counter;
    counter["cold"] = 0;
    counter["normal"] = 0;
    counter["hot"] = 0;

    for (int temp=-5;temp<=40;temp+=5)
        counter[classify(temp)]++;

    for (const auto& pair : counter)
        std::cout << pair.first << " : " << pair.second << std::endl;
}