#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_var_;
    bool finished_ = false;
    size_t max_size_;

public:
    explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return queue_.size() < max_size_; });
        queue_.push(std::move(item));
        cond_var_.notify_all();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return !queue_.empty() || finished_; });
        
        if (queue_.empty() && finished_) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();
        cond_var_.notify_all();
        return item;
    }

    void set_finished() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        cond_var_.notify_all();
    }
};

#endif // THREAD_SAFE_QUEUE_H
