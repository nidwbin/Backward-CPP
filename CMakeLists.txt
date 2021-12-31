cmake_minimum_required(VERSION 3.22)
project(backward)

set(CMAKE_CXX_STANDARD 14)

add_executable(backward main.cpp lib/Tensor.cpp lib/Tensor.h lib/Functions.cpp lib/Functions.h)
