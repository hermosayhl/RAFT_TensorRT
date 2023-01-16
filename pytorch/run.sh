python export.py

# 简化模型
python -m onnxsim ../onnxruntime/RAFT.onnx ../onnxruntime/RAFT_simplified.onnx

# 测试性能
trtexec --onnx=../onnxruntime/RAFT_simplified.onnx --verbose --minShapes=image1:1x3x256x256,image2:1x3x256x256 --optShapes=image1:1x3x440x1024,image2:1x3x440x1024 --maxShapes=image1:1x3x768x1024,image2:1x3x768x1024 --saveEngine=../tensorrt/engine/RAFT.plan

# 随机数导致的这么大误差
polygraphy run ../onnxruntime/RAFT_simplified.onnx --onnxrt --trt --workspace 1000000000 --save-engine=../tensorrt/engine/RAFT2.plan --atol 0.1 --verbose --trt-min-shapes 'image1:[1,3,256,256]' 'image2:[1,3,256,256]' --trt-opt-shapes 'image1:[1,3,440,1024]' 'image2:[1,3,440,1024]' --trt-max-shapes 'image1:[1,3,768,1024]' 'image2:[1,3,768,1024]' --input-shapes 'image1:[1,3,440,1024]' 'image2:[1,3,440,1024]' --val-range 'image1:[0,255]' 'image2:[0,255]'

# 查看模型
polygraphy inspect model "../onnxruntime/RAFT_simplified.onnx" --display-as=trt
polygraphy inspect model ../onnxruntime/RAFT2.plan --display-as=trt  --model-type engine

# polygraphy 也可以简化模型
polygraphy surgeon sanitize ../onnxruntime/RAFT_simplified.onnx ../onnxruntime/RAFT_sanitized.onnx --fold-constants

# 测测 fp16
polygraphy run ../onnxruntime/RAFT_simplified.onnx --fp16 --onnxrt --trt --workspace 1000000000 --save-engine=../tensorrt/engine/RAFT_fp16.plan --atol 0.1 --verbose --trt-min-shapes 'image1:[1,3,256,256]' 'image2:[1,3,256,256]' --trt-opt-shapes 'image1:[1,3,440,1024]' 'image2:[1,3,440,1024]' --trt-max-shapes 'image1:[1,3,768,1024]' 'image2:[1,3,768,1024]' --input-shapes 'image1:[1,3,440,1024]' 'image2:[1,3,440,1024]' --val-range 'image1:[0,255]' 'image2:[0,255]'
