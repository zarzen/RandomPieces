#include <queue>
#include <vector>
#include <thread>
#include <iostream>
#include <mutex>

void getItem(std::queue<int>& tasks, std::mutex& mtx) {
  bool empty = true;
  while (empty) {
    std::lock_guard<std::mutex> lk(mtx);
    empty = tasks.empty();
  }
  std::lock_guard<std::mutex> lk(mtx);
  printf("get %d\n", tasks.front());
}

int main() {
  int n = 2;
  std::vector<std::thread> thds;
  std::vector<std::queue<int>> queues;
  std::vector<std::mutex> mtxs(n);

  for (int i = 0; i < n; ++i) {
    queues.push_back(std::queue<int>());
    thds.emplace_back(getItem, std::ref(queues[i]), std::ref(mtxs[i]));
  }

  for (int i = 0; i < n; ++i) {
    std::lock_guard<std::mutex> lk(mtxs[i]);
    queues[i].push(i);
    printf("pushed to %d with %d\n", i, i);
  }

  for (auto& t:thds) {
    t.join();
  }
}