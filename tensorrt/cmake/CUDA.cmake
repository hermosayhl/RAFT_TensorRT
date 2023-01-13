
# CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
set(ThirdPartyLIBs ${ThirdPartyLIBs} cudart)
