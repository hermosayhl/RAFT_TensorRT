
# TensorRT
set(TensorRT_DIR D:/HPC/tools/TensorRT/TensorRT-8.5.1.7)
include_directories(${TensorRT_DIR}/include)
include_directories(${TensorRT_DIR}/samples/common)
set(TensorRT_LIBRARY_DIRS ${TensorRT_DIR}/lib)
link_directories(TensorRT_LIBRARY_DIRS)
# 这里得填全名, 否则会报错
set(ThirdPartyLIBs ${ThirdPartyLIBs} 
	${TensorRT_LIBRARY_DIRS}/nvinfer.lib
	${TensorRT_LIBRARY_DIRS}/nvonnxparser.lib)