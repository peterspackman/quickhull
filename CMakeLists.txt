cmake_minimum_required(VERSION 3.21)
project(quickhull)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(BUILD_TESTING "Build tests" ON)
endif() 

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_library(quickhull "${CMAKE_CURRENT_SOURCE_DIR}/src/quickhull.cpp")
target_link_libraries(quickhull PUBLIC Eigen3::Eigen)
target_include_directories(quickhull PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_compile_features(quickhull PUBLIC cxx_std_17)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
enable_testing()

set(TEST_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_quickhull.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/catch_amalgamated.cpp"
)

add_executable(tests ${TEST_FILES})
target_link_libraries(tests PUBLIC quickhull)
add_test(NAME quickhull_tests COMMAND tests)

endif()
