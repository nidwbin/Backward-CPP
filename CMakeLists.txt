cmake_minimum_required(VERSION 3.22)
project(backward)

set(CMAKE_CXX_STANDARD 14)

add_executable(backward main.cpp lib/Tensor.cpp lib/Tensor.h lib/Module.cpp lib/Module.h lib/Linear.cpp lib/Linear.h lib/Sigmoid.cpp lib/Sigmoid.h lib/MSE.cpp lib/MSE.h)
