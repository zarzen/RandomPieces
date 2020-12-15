#include <iostream>
#include <vector>

class A {
  std::vector<int> self_v;

 public:
  void assignVector(std::vector<int>& v) { self_v = std::move(v); }

  void printSelfVector() {
    for (auto v : self_v) {
      printf("%d\n", v);
    }
  }
};

int main(int argc, char* argv[]) {
  std::vector<int> v1;
  for (int i = 0; i < 10; i++) {
    v1.push_back(i);
  }

  A a;
  a.assignVector(v1);

  a.printSelfVector();
  printf("printSelfVector end\n");
  for (auto x : v1) {
    printf("%d\n", x);
  }
}