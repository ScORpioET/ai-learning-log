#include <queue>
#include <mutex>
#include <atomic>
#include <thread>
#include <iostream>
#include <condition_variable>

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

int n = 10;
std::mutex mtx;
std::condition_variable cv;
std::queue<std::unique_ptr<Sensor>> q;
std::atomic<bool> producer_done{false};

void producer(){
    for(int i=0;i<n;i++){
        std::unique_lock<std::mutex> lock(mtx);
        q.push(std::make_unique<Sensor>((i+1)*10));
        lock.unlock();
        cv.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    producer_done = true;
    cv.notify_one();
}

void consumer(){
    int i = 0;
    while(true){
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] {return producer_done || !q.empty();});

        if (producer_done && q.empty()){
            lock.unlock();
            break;
        }
        
        std::cout << "Sensor" << ++i << " is triggerd : " << q.front()->isTriggered(67.0) << std::endl;
        q.pop();
        lock.unlock();
    }
}

int main(){
    std::thread p(producer);
    std::thread c(consumer);

    p.join();
    c.join();
}