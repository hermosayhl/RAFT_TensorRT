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
sys.path.append("../pytorch/core/utils")
import flow_viz


def cv_show(image, message="crane"):
	cv2.imshow(message, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

cv_write = lambda x, y: cv2.imwrite(x, y, [cv2.IMWRITE_PNG_COMPRESSION, 0])


# 读取两张图象
image1 = cv2.imread("../pytorch/demo-frames/frame_0016.png")
image2 = cv2.imread("../pytorch/demo-frames/frame_0017.png")

# onnx 支持边长倍数为 8 的图片
assert image1.shape == image2.shape, "形状要一致"
image1 = numpy.pad(image1, [(2, 2), (0, 0), (0, 0)], mode="reflect")
image2 = numpy.pad(image2, [(2, 2), (0, 0), (0, 0)], mode="reflect")
print("image1  :  ", image1.shape)
print("image2  :  ", image2.shape)

# 把图像从 numpy.uint8、BGR、HWC 转化成 numpy.float32、RGB、BCHW, B=1
def convert_to_tensor(x):
	x_tensor = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	x_tensor = numpy.ascontiguousarray(x_tensor.transpose(2, 0, 1))
	x_tensor = numpy.expand_dims(x_tensor, axis=0)
	return x_tensor.astype("float32")

image1_tensor = convert_to_tensor(image1)
image2_tensor = convert_to_tensor(image2)

# 加载推理引擎
import onnxruntime
engine = onnxruntime.InferenceSession("./RAFT_simplified.onnx", providers=["CPUExecutionProvider"])

# 开始推理
[flow] = engine.run(["flow"], {"image1": image1_tensor, "image2": image2_tensor})
print("flow  :  ", flow.shape)

# 展示
flow = numpy.ascontiguousarray(flow[0].transpose(1, 2, 0))
flow_visualize = flow_viz.flow_to_image(flow, convert_to_bgr=True)
cv_write("flow.png", flow_visualize)
cv_show(flow_visualize)

# 保存
numpy.save("./flow.npy", flow)

