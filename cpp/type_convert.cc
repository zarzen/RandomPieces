#include <iostream>

int main() {
  int i = -1;
  int b = 3;
  uint64_t s[2];
  s[0] = (uint64_t)i;
  s[1] = (uint64_t)b;
  std::cout<< s[0] << " , " << s[1];
  return 0;
}