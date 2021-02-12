cmake_minimum_required(VERSION 3.10)
project(randomCuda LANGUAGES CUDA CXX)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)
set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE MATCHES Release)
  message("-- log level info")
  add_definitions(-DLOG_LEVEL=2)
else()
  # debug
  message("-- enable debug logging")
  add_definitions(-DLOG_LEVEL=3)
endif()

add_executable(interactTest hostInteractDev.cc kernels.cu kernels.h)
set_property(TARGET interactTest PROPERTY CUDA_ARCHITECTURES 70 72 75)
set_target_properties(interactTest
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(interactTest PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(interactTest pthread)

add_executable(summationTest testSummation.cc kernels.cu kernels.h)
set_property(TARGET summationTest PROPERTY CUDA_ARCHITECTURES 70 72 75)
set_target_properties(summationTest
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(summationTest PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(matrixAdd matrix_add.cu)
set_property(TARGET matrixAdd PROPERTY CUDA_ARCHITECTURES 70 72 75)
set_target_properties(matrixAdd
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(matrixAdd PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})


add_executable(dataAssign data_assign.cc kernels.cu kernels.h)
set_property(TARGET dataAssign PROPERTY CUDA_ARCHITECTURES 70 72 75)
set_target_properties(dataAssign
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(dataAssign PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
