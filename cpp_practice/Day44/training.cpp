#include <iostream>
#include <string>
#include <vector>



int add(double x, double y){
    return x + y;
}

void addOneByValue(int x) { x = x + 1; }
void addOneByPointer(int* x) { *x = *x + 1; }
void addOneByReference(int& x) { x = x + 1; }

int main() {
    int a = 7.0;
    double b = 2;

    std::cout << "a / b = " << (a/b) << std::endl;

    std::cout << "x + y = " << add(double(3.5), double(2.5)) << std::endl;

    for(int i=0;i<10;i++)
        if(i%2==0)
            std::cout << i << std::endl;

    int i = 0;
    while(i<10){
        if(i%2==0){
            std::cout << i << std::endl;

        }
        i++;
    }


    int c = 10, d = 10, f = 10;

    addOneByValue(c);       // 直接傳 a,函式要的是 int,傳值進去複製一份
    addOneByPointer(&d);    // 傳 &b,因為函式要的是 int*(指標),你要先把 b 的「位址」取出來給它
    addOneByReference(f);   // 直接傳 c,函式要的是 int&,語法上看起來跟傳值一樣

    std::cout << "c = " << c << ", d = " << d << ", f = " << f << std::endl;


    // int* p = nullptr;
    // if (p != nullptr) {
    //     std::cout << *p << std::endl;
    // } else {
    //     std::cout << "p is null, skip dereference" << std::endl;
    // }


    int* p = new int(42);
    std::cout << "before delete: " << *p << std::endl;

    delete p;
    std::cout << "after delete: " << *p << std::endl;
    

    

}