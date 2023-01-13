# pytorch
import os
import sys
import time
# 3rd party
import cv2
import numpy
import ctypes
from numpy.ctypeslib import ndpointer
# self


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 读取一次光流
flow = numpy.load("../../onnxruntime/flow.npy")

# # 用 RAFT 提供的可视化代码展示
# sys.path.append("../../pytorch/core/utils")
# import flow_viz
# cv_show(flow_viz.flow_to_image(flow, convert_to_bgr=True))

# 直接模型输出 1x2xHxW, 接到这里
flow = numpy.expand_dims(numpy.ascontiguousarray(flow.transpose(2, 0, 1)), axis=0)
print("flow  :  ", flow.shape)

# 编译 C++ 代码
visualize_lib_path = "./visualize.so"
os.system("g++ -fPIC -shared -O2 ./visualize.cpp -o {}".format(visualize_lib_path))

# 加载动态库
visualize_lib = ctypes.cdll.LoadLibrary(visualize_lib_path)

# 转换
_, _, height, width = flow.shape
result = numpy.zeros((height, width, 3), dtype="uint8")
visualize_lib.flow_to_image(
	result.ctypes.data_as(ctypes.c_char_p),
	flow.ctypes.data_as(ctypes.c_char_p),
	height,
	width
)
cv_show(result)