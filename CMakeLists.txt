
cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER "g++")
#set(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-13")

# Set the project name, version, and primary programming language(s) 
project(WangLandauNeuro VERSION 0.1 LANGUAGES CXX)

# Set policy to silence Warnings about timestamps of downloaded files.
# Review occasionally if it may be set to NEW
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 OLD)
endif()

set(INCLUDE_DIR "./inc")
set(SOURCE_DIR "./src/CXX")
set(LIBRARY_DIR "./lib")

# Define minimum C++ version for Mac
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define colors for bringing attention to OMP & MPI ON/OFF
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(CReset  "${Esc}[m")
  set(R       "${Esc}[1;31m")
  set(G       "${Esc}[1;32m")
endif()

add_executable(WL.x src/CXX/main.cpp)
add_executable(Brain.x src/CXX/neuron.cpp)
add_library(utilLib SHARED src/CXX/utils.cpp)
add_library(FFNNLib SHARED lib/FFNN.cpp src/CXX/utils.cpp)

FIND_PACKAGE(Gnuplot REQUIRED)
IF(GNUPLOT_FOUND)
  message("${G} #       ------- GNUPlot SUPPORT IS ON       #${CReset}")
  add_definitions(-DMyGNUPlot="${PROJECT_SOURCE_DIR}/src/Others/plotHistogram.plt")
ELSE()
  message("${G} #               GNUPlot SUPPORT IS ON       #${CReset}")
ENDIF()

option(USE_OMP "Enable OpenMP calls" OFF)
IF(USE_OMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  IF(OpenMP_FOUND)
    target_link_libraries(FFNNLib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(WL.x PUBLIC -fopenmp)
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      target_compile_options(WL.x PUBLIC -Xclang)
    ENDIF()
    message("${G} #       ------------ OPENMP SUPPORT IS ON   #${CReset}")
  ELSE()
    message("${R} #       ------------- OPENMP SUPPORT IS OFF #${CReset}")
  ENDIF()
ELSE()
  message(  "${R} #           OPENMP SUPPORT IS OFF           #${CReset}")
ENDIF()

include_directories(inc)
target_link_libraries(WL.x FFNNLib)
target_link_libraries(Brain.x utilLib)

option(TEST "Enable Unit Tests" OFF)
IF(TEST)
  enable_testing()
  enable_language(CXX)

  set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
  set(DEBUG 3)
  include(GTest)

  add_executable(test_utils test/CXX/test_utils.cpp)
  target_include_directories(test_utils PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(test_utils PRIVATE GTest::GTestMain GTest::GTest utilLib)
  add_test(NAME Utils COMMAND test_utils)

  # Link test_utils with OpenMP
  IF(USE_OMP)
    target_link_libraries(test_utils PRIVATE GTest::GTestMain GTest::GTest utilLib OpenMP::OpenMP_CXX)
    target_compile_options(test_utils PUBLIC -fopenmp)
    # Fix OpenMP compilation when the compiler is Clang
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
      target_compile_options(test_utils PUBLIC -Xclang)
    ENDIF()
  ELSE()
    target_link_libraries(test_utils PRIVATE GTest::GTestMain GTest::GTest utilLib)
  ENDIF()

  add_executable(test_WL test/CXX/test_WL.cpp)
  target_include_directories(test_WL PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(test_WL PRIVATE GTest::GTestMain GTest::GTest FFNNLib)
  add_test(NAME WL COMMAND test_WL)

  add_executable(test_neuron test/CXX/test_neuron.cpp)
  target_include_directories(test_neuron PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(test_neuron PRIVATE GTest::GTestMain GTest::GTest utilLib)
  add_test(NAME Brain COMMAND test_neuron)
ENDIF()
