# 项目
PROJECT(RAFT_TRT)
cmake_minimum_required(VERSION 3.10)

# 设置可执行文件目录
set(WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 编译参数
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_definitions(-w)

# 添加头文件
message(STATUS "PROJECT  ${PROJECT_SOURCE_DIR}/include")
include_directories(SYSTEM
	${PROJECT_SOURCE_DIR}/include)

# 添加外部依赖
set(ThirdPartyLIBs "")
set(ThirdPartySource "")
include(${PROJECT_SOURCE_DIR}/cmake/OpenCV.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/CUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/TensorRT.cmake)

# 添加可执行文件
add_executable(raft_estimator 
	src/inference.cpp 
	${ThirdPartySource}
)

# 链接库
target_link_libraries(raft_estimator ${ThirdPartyLIBs})