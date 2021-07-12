#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

template <typename T>
class ThdSafeQueue {
 public:
  ThdSafeQueue(){};
  ~ThdSafeQueue(){};
  bool isEmpty(){return this->queue_.empty();};

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  void push(T new_value) {
    mu_.lock();
    queue_.push(std::move(new_value));
    mu_.unlock();
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  void pop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this] { return !queue_.empty(); });
    *value = std::move(queue_.front());
    queue_.pop();
  }

  // async pop
  bool apop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) return false;
    *value = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  void wait() {
    if (queue_.empty()) {
      std::unique_lock<std::mutex> lk(mu_);
      cond_.wait(lk, [this] { return !queue_.empty(); });
    }
  }

 private:
  mutable std::mutex mu_;
  std::queue<T> queue_;
  std::condition_variable cond_;
};
