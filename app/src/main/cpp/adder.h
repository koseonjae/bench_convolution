#pragma once

#include <vector>
#include <string>

std::vector<int> generateRandomImage();

class Timer {
 public:
  Timer(std::string task_name);

  ~Timer();

 private:
  std::string task_name;
  std::chrono::time_point<std::chrono::steady_clock> start;
};