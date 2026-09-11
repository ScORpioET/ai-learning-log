#include <mutex>
#include <queue>
#include <thread>
#include <memory>
#include <iostream>
#include <condition_variable>


int counter = 0;   // 共享的全域變數
std::mutex mtx;
std::queue<int> q;
std::condition_variable cv;

class Sensor{
    public:
        Sensor(double t) : threshold(t) {
            std::cout << "Sensor 建構, threshold = " << threshold << std::endl;
        }
        ~Sensor(){
            std::cout << "Sensor 解構, threshold = " << threshold << std::endl;
        }

    private:
        double threshold;
};

void worker(int id) {
    std::cout << "thread " << id << " 開始工作" << std::endl;
}


void increment() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx);
        counter++;   // 沒有任何保護
    }
}


void producer(){
    for (int i=0;i<5;i++){
        // std::lock_guard<std::mutex> lock(mtx);
        std::unique_lock<std::mutex> lock(mtx);
        q.push(i);
        std::cout << "produced " << i << std::endl;
        lock.unlock();
        cv.notify_one();
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}

void consumer() {
    for (int i=0;i<5;i++){
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] {return !q.empty();});
        int val = q.front();
        q.pop();
        std::cout << "consumed " << val << std::endl;
        lock.unlock();
    }
}

int main() {
    {
        std::unique_ptr<Sensor> p = std::make_unique<Sensor>(20.0);
        std::cout << "scope 內,p 還活著" << std::endl;
    }
    std::cout << "scope  外" << std::endl;


    std::cout << "-----------------------------------------" << std::endl;


    std::shared_ptr<Sensor> p1 = std::make_shared<Sensor>(20.0);
    std::cout << "use_count = " << p1.use_count() << std::endl;

    {
        std::shared_ptr<Sensor> p2 = p1;
        std::cout << "use_count = " << p1.use_count() << std::endl;
    }

    std::cout << "use_count = " << p1.use_count() << std::endl;  // 1
    std::cout << "p1 還活著,還能用" << std::endl;

    std::cout << "-----------------------------------------" << std::endl;


    std::thread t1(worker, 1);
    std::thread t2(worker, 2);

    std::cout << "main thread 繼續做自己的事" << std::endl;

    t1.join();
    t2.join();

    std::cout << "全部 thread 都結束了" << std::endl;

    std::cout << "-----------------------------------------" << std::endl;


    std::thread t3(increment);
    std::thread t4(increment);
    t3.join();
    t4.join();
    std::cout << "counter = " << counter << std::endl;

    std::cout << "-----------------------------------------" << std::endl;

    std::thread t5(producer);
    std::thread t6(consumer);
    t5.join();
    t6.join();
}