
# 寻找 OpenCV
set(OpenCV_DIR D:/environments/C++/OpenCV/opencv-msvc/build)
find_package(OpenCV REQUIRED)
# 添加 OpenCV 头文件路径
include_directories(${OpenCV_INCLUDE_DIRS})
# 添加 OpenCV 动态库路径
# 电脑属性环境变量 -> add D:\environments\C++\OpenCV\opencv-msvc\build\x64\vc15\bin
link_directories(${OpenCV_LIBRARY_DIRS})
# 链接动态库
set(ThirdPartyLIBs ${ThirdPartyLIBs} ${OpenCV_LIBS})
# 输出信息
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    lib path: ${OpenCV_LIB_DIRS}")
message(STATUS "    Debug libraries: ${OpenCV_Debug_LIBS}")
message(STATUS "    Release libraries: ${OpenCV_Release_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")