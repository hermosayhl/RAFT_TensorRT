// C++
#include <assert.h>
#include <filesystem>
// Third party ==> CUDA、OpenCV、TensorRT
#include "utils_cuda.hpp"
#include "utils_opencv.hpp"
#include "utils_tensorrt.hpp"
// self 
#include "utils_visualize.hpp"




// 测试 opencv 环境
void test_opencv_environment() {
	auto image1 = cv::imread("../../pytorch/demo-frames/frame_0016.png");
	if (image1.empty()) {
		std::cout << "read failure!\n";
		return;
	}
	cv_show(image1);
}

// 测试 cuda 环境
void test_cuda_environment() {
	AutoCudaMallocPointer<float> data;
	data.allocate(256 * 256, CUDA_DATA_TYPE::NORMAL_MEMORY, "test_environment");
}

// 测试 tensorrt 环境
void test_tensorrt_environment() {
	/* TensorRT 引擎路径 */
	const std::string tensorrt_engine_path("../engine/RAFT.plan");
	assert(std::filesystem::exists(tensorrt_engine_path) && "engine file doesn't exist!");

	/* 反序列化得到 tensorrt 引擎 */
	ICudaEngine* engine = deserialize(tensorrt_engine_path);
}




int main() {
	// 查看当前目录
	std::cout << "working dir==> " << std::filesystem::current_path().string() << "\n";
	
	/* TensorRT 引擎路径 */
	const std::string tensorrt_engine_path("../engine/RAFT_fp32.plan");
	assert(std::filesystem::exists(tensorrt_engine_path) && "engine file doesn't exist!");
	
	/* 反序列化得到 tensorrt 引擎 */
	ICudaEngine* engine = deserialize(tensorrt_engine_path);
	if (engine == nullptr) {
		return -1;
	}

	/* 创建上下文 GPU 进程 */
	IExecutionContext* context = engine->createExecutionContext();

	/* 找到推理模型中输入张量的个数, 确认模型 */
	int32_t tensors_count = engine->getNbBindings();
	int32_t inputs_count = 0;
	for (int i = 0; i < tensors_count; ++i) {
		std::cout << i << ", " << engine->getBindingName(i) << "\n";
		inputs_count += engine->bindingIsInput(i);
	}
	std::cout << "tensors_count  =  " << tensors_count << "\n";
	std::cout << "inputs_count   =  " << inputs_count  << "\n";

	/* 绑定模型输入的形状(后续这里得加个判断, 获取最大维度、最小维度, 对输入做规范); sintel 数据这里可以改成 440x1024; real 真实场景数据改成 448 * 600 */
	constexpr int32_t batch_size     = 1;
	constexpr int32_t input_channel  = 3;
	constexpr int32_t image_height   = 440;
	constexpr int32_t image_width    = 1024;
	constexpr int32_t output_channel = 2;
	context->setBindingDimensions(0, Dims4{batch_size, input_channel,  image_height, image_width});
	context->setBindingDimensions(1, Dims4{batch_size, input_channel,  image_height, image_width});
	
	/* 计算输入张量, 输出张量的元素个数, 在 RAFT 模型中默认都是 float32；输出张量的形状可以通过 context->getBindingDimension(i)获取 */
	constexpr int32_t image1_element_count = batch_size * input_channel * image_height * image_width;
	constexpr int32_t image2_element_count = batch_size * input_channel * image_height * image_width;
	constexpr int32_t output_element_count = batch_size * output_channel * image_height * image_width;

	/* 在 CPU 端准备输入和输出的内存 */
	AutoCudaMallocPointer<float> image1_buffer_cpu(image1_element_count, CUDA_DATA_TYPE::HOST_MEMORY, "image1_buffer_cpu");
	AutoCudaMallocPointer<float> image2_buffer_cpu(image2_element_count, CUDA_DATA_TYPE::HOST_MEMORY, "image2_buffer_cpu");
	AutoCudaMallocPointer<float> flow_buffer_cpu(output_element_count,   CUDA_DATA_TYPE::HOST_MEMORY, "flow_buffer_cpu");

	/* 在 GPU 端准备输入和输出的内存 */
	AutoCudaMallocPointer<float> image1_buffer_gpu(image1_element_count, CUDA_DATA_TYPE::NORMAL_MEMORY, "image1_buffer_gpu");
	AutoCudaMallocPointer<float> image2_buffer_gpu(image2_element_count, CUDA_DATA_TYPE::NORMAL_MEMORY, "image2_buffer_gpu");
	AutoCudaMallocPointer<float> flow_buffer_gpu(output_element_count,   CUDA_DATA_TYPE::NORMAL_MEMORY, "flow_buffer_gpu");

	/* 读取两帧图像 */
	auto image1 = cv::imread("../images/input/frame_0016.png");
	auto image2 = cv::imread("../images/input/frame_0017.png");
	if (image1.empty() || image2.empty()) {
		std::cout << "image1 or image2 is empty! please check!\n";
		return -2;
	}

	/* 图像大小必须和模型输入输出一致, 这里进行简单的 resize! */
	cv::resize(image1, image1, {image_width, image_height}, cv::INTER_LINEAR);
	cv::resize(image2, image2, {image_width, image_height}, cv::INTER_LINEAR);


	/* 把图像转换成 RGB 序, 存储进 CPU 的输入 buffer 中 */
	auto convert_to_tensor = [image_height, image_width, input_channel](unsigned char* image_ptr, float* tensor_ptr) {
		/* HWC→CHW  */
		const int32_t length = image_height * image_width;
		for (int32_t c = 0; c < input_channel; ++c) {
			auto res_ptr = tensor_ptr + length * (input_channel - 1 - c);
			for (int32_t i = 0; i < length; ++i) {
				res_ptr[i] = float(image_ptr[3 * i + c]);
			}
		}
	};
	convert_to_tensor(image1.ptr<uchar>(), image1_buffer_cpu.get_pointer());
	convert_to_tensor(image2.ptr<uchar>(), image2_buffer_cpu.get_pointer());

	/* 把输入数据从 cpu 传送到 gpu */
	ck(cudaMemcpy(image1_buffer_gpu.get_pointer(), image1_buffer_cpu.get_pointer(), image1_element_count * sizeof(float), cudaMemcpyHostToDevice));
	ck(cudaMemcpy(image2_buffer_gpu.get_pointer(), image2_buffer_cpu.get_pointer(), image1_element_count * sizeof(float), cudaMemcpyHostToDevice));
	
	/* 准备一个指针数组, 包含输入和输出的 GPU 内存指针, 方便 tensorrt 推理 */
	std::vector<void*> gpu_tensors({
		(void*)image1_buffer_gpu.get_pointer(),
		(void*)image2_buffer_gpu.get_pointer(),
		(void*)flow_buffer_gpu.get_pointer(),
	});

	/* 开始推理 */
	cuda_timer([&]() {
		/* 目前测试的推理时间有问题 */
		auto infer_sign = context->executeV2(gpu_tensors.data());
		if (infer_sign == false) {
			std::cerr << "context->executeV2 failed\n";
			return -3;
		}},
		"fp16"
	);
	

	/* 把推理结果从 GPU 传送到 CPU 的 buffer */
	ck(cudaMemcpy(flow_buffer_cpu.get_pointer(), flow_buffer_gpu.get_pointer(), output_element_count * sizeof(float), cudaMemcpyDeviceToHost));


	/* 解析 flow_buffer_cpu 的内容, 展示 */
	cv::Mat flow_visualize(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));
	flow_to_image_inplementation(
		flow_visualize.data,
		flow_buffer_cpu.get_pointer(),
		image_height,
		image_width
	);

	/* 可视化 */
	cv_show(flow_visualize);

	/* 保存 */
	cv::imwrite("../images/output/sintel.png", flow_visualize, { cv::IMWRITE_PNG_COMPRESSION, 0});

	return 0;
}