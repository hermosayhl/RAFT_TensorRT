
# CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
set(ThirdPartyLIBs ${ThirdPartyLIBs} cudart)
# 输出信息
message(STATUS "    CUDA version: ${CUDA_VERSION}")
message(STATUS "    CUDA lib path: ${CUDA_LIBRARY_DIRS}")
message(STATUS "    CUDA include path: ${CUDA_INCLUDE_DIRS}")
