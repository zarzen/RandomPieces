#include <unordered_map>
#include <vector>
#include <iostream>

using namespace std;

int main() {
  vector<int> sizes;
  sizes.push_back(100);
  sizes.push_back(200);
  // sizes.push_back(100);

  unordered_map<int, int> counter;
  for (auto& s : sizes) {
    if (counter.count(s) == 0) {
      counter[s] = 1;
    } else {
      counter[s]++;
    }
  }
  int max_count = 0;
  int major_size = 0;
  for (auto it : counter ){
    printf("%d, %d\n", it.first, it.second);
    if (it.second > max_count) {
      major_size = it.first;
      max_count = it.second;
    }
  }

  printf("major size %d, %d\n", major_size, max_count);
}