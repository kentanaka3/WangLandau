#include "main.hpp"

int main(int argc, char *argv[]) {
  WL * myWL = set_problem(argv[1], argv[2]);
  myWL->run();
  delete myWL;
  return 0;
}