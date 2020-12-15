#include <iostream>
#include <memory>
#include <functional>
using std::shared_ptr;


struct A {

  shared_ptr<int> v;

  std::function<void()> lazy_init;
};

shared_ptr<int> retPtr(shared_ptr<int> a) {
  return a;
}

int main(int argc, char* argv[]) {
  shared_ptr<int> p = std::make_shared<int>(10);
  shared_ptr<A> a = std::make_shared<A>();
  a->lazy_init = [&, p, a] () { a->v = retPtr(p); };

  a->lazy_init();
  printf("a get v %d \n", *a->v);
  *a->v = 32;
  printf("modified a->v %d, p %d \n", *a->v, *p);

  std::function<void()> f1 = [](){};
  f1();
  std::function<void()> f2;
  if (f2 == nullptr) {
    printf("f2 is null");
  }

  return 0;
}