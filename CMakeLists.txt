cmake_minimum_required(VERSION 3.11)
project(QuickHullTests)

find_package(Eigen3 REQUIRED)

set(SOURCE_FILES
  ${CMAKE_SOURCE_DIR}/tests/QuickHullTests.cpp
  ${CMAKE_SOURCE_DIR}/tests/main.cpp
  ${CMAKE_SOURCE_DIR}/src/quickhull.cpp
  )

add_executable(run_tests ${SOURCE_FILES})
target_compile_features(run_tests PUBLIC cxx_std_17)
target_link_libraries(run_tests PUBLIC Eigen3::Eigen)
target_include_directories(run_tests PUBLIC "${CMAKE_SOURCE_DIR}/include")
