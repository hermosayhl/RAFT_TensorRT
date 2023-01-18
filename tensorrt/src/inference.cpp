// C++
#include <assert.h>
#include <filesystem>
// Third party ==> CUDA��OpenCV��TensorRT
#include "utils_cuda.hpp"
#include "utils_opencv.hpp"
#include "utils_tensorrt.hpp"
// self 
#include "utils_visualize.hpp"




// ���� opencv ����
void test_opencv_environment() {
	auto image1 = cv::imread("../../pytorch/demo-frames/frame_0016.png");
	if (image1.empty()) {
		std::cout << "read failure!\n";
		return;
	}
	cv_show(image1);
}

// ���� cuda ����
void test_cuda_environment() {
	AutoCudaMallocPointer<float> data;
	data.allocate(256 * 256, CUDA_DATA_TYPE::NORMAL_MEMORY, "test_environment");
}

// ���� tensorrt ����
void test_tensorrt_environment() {
	/* TensorRT ����·�� */
	const std::string tensorrt_engine_path("../engine/RAFT.plan");
	assert(std::filesystem::exists(tensorrt_engine_path) && "engine file doesn't exist!");

	/* �����л��õ� tensorrt ���� */
	ICudaEngine* engine = deserialize(tensorrt_engine_path);
}




int main() {
	// �鿴��ǰĿ¼
	std::cout << "working dir==> " << std::filesystem::current_path().string() << "\n";
	
	/* TensorRT ����·�� */
	const std::string tensorrt_engine_path("../engine/RAFT_fp32.plan");
	assert(std::filesystem::exists(tensorrt_engine_path) && "engine file doesn't exist!");
	
	/* �����л��õ� tensorrt ���� */
	ICudaEngine* engine = deserialize(tensorrt_engine_path);
	if (engine == nullptr) {
		return -1;
	}

	/* ���������� GPU ���� */
	IExecutionContext* context = engine->createExecutionContext();

	/* �ҵ�����ģ�������������ĸ���, ȷ��ģ�� */
	int32_t tensors_count = engine->getNbBindings();
	int32_t inputs_count = 0;
	for (int i = 0; i < tensors_count; ++i) {
		std::cout << i << ", " << engine->getBindingName(i) << "\n";
		inputs_count += engine->bindingIsInput(i);
	}
	std::cout << "tensors_count  =  " << tensors_count << "\n";
	std::cout << "inputs_count   =  " << inputs_count  << "\n";

	/* ��ģ���������״(��������üӸ��ж�, ��ȡ���ά�ȡ���Сά��, ���������淶); sintel ����������Ըĳ� 440x1024; real ��ʵ�������ݸĳ� 448 * 600 */
	constexpr int32_t batch_size     = 1;
	constexpr int32_t input_channel  = 3;
	constexpr int32_t image_height   = 440;
	constexpr int32_t image_width    = 1024;
	constexpr int32_t output_channel = 2;
	context->setBindingDimensions(0, Dims4{batch_size, input_channel,  image_height, image_width});
	context->setBindingDimensions(1, Dims4{batch_size, input_channel,  image_height, image_width});
	
	/* ������������, ���������Ԫ�ظ���, �� RAFT ģ����Ĭ�϶��� float32�������������״����ͨ�� context->getBindingDimension(i)��ȡ */
	constexpr int32_t image1_element_count = batch_size * input_channel * image_height * image_width;
	constexpr int32_t image2_element_count = batch_size * input_channel * image_height * image_width;
	constexpr int32_t output_element_count = batch_size * output_channel * image_height * image_width;

	/* �� CPU ��׼�������������ڴ� */
	AutoCudaMallocPointer<float> image1_buffer_cpu(image1_element_count, CUDA_DATA_TYPE::HOST_MEMORY, "image1_buffer_cpu");
	AutoCudaMallocPointer<float> image2_buffer_cpu(image2_element_count, CUDA_DATA_TYPE::HOST_MEMORY, "image2_buffer_cpu");
	AutoCudaMallocPointer<float> flow_buffer_cpu(output_element_count,   CUDA_DATA_TYPE::HOST_MEMORY, "flow_buffer_cpu");

	/* �� GPU ��׼�������������ڴ� */
	AutoCudaMallocPointer<float> image1_buffer_gpu(image1_element_count, CUDA_DATA_TYPE::NORMAL_MEMORY, "image1_buffer_gpu");
	AutoCudaMallocPointer<float> image2_buffer_gpu(image2_element_count, CUDA_DATA_TYPE::NORMAL_MEMORY, "image2_buffer_gpu");
	AutoCudaMallocPointer<float> flow_buffer_gpu(output_element_count,   CUDA_DATA_TYPE::NORMAL_MEMORY, "flow_buffer_gpu");

	/* ��ȡ��֡ͼ�� */
	auto image1 = cv::imread("../images/input/frame_0016.png");
	auto image2 = cv::imread("../images/input/frame_0017.png");
	if (image1.empty() || image2.empty()) {
		std::cout << "image1 or image2 is empty! please check!\n";
		return -2;
	}

	/* ͼ���С�����ģ���������һ��, ������м򵥵� resize! */
	cv::resize(image1, image1, {image_width, image_height}, cv::INTER_LINEAR);
	cv::resize(image2, image2, {image_width, image_height}, cv::INTER_LINEAR);


	/* ��ͼ��ת���� RGB ��, �洢�� CPU ������ buffer �� */
	auto convert_to_tensor = [image_height, image_width, input_channel](unsigned char* image_ptr, float* tensor_ptr) {
		/* HWC��CHW  */
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

	/* ���������ݴ� cpu ���͵� gpu */
	ck(cudaMemcpy(image1_buffer_gpu.get_pointer(), image1_buffer_cpu.get_pointer(), image1_element_count * sizeof(float), cudaMemcpyHostToDevice));
	ck(cudaMemcpy(image2_buffer_gpu.get_pointer(), image2_buffer_cpu.get_pointer(), image1_element_count * sizeof(float), cudaMemcpyHostToDevice));
	
	/* ׼��һ��ָ������, �������������� GPU �ڴ�ָ��, ���� tensorrt ���� */
	std::vector<void*> gpu_tensors({
		(void*)image1_buffer_gpu.get_pointer(),
		(void*)image2_buffer_gpu.get_pointer(),
		(void*)flow_buffer_gpu.get_pointer(),
	});

	/* ��ʼ���� */
	cuda_timer([&]() {
		/* Ŀǰ���Ե�����ʱ�������� */
		auto infer_sign = context->executeV2(gpu_tensors.data());
		if (infer_sign == false) {
			std::cerr << "context->executeV2 failed\n";
			return -3;
		}},
		"fp16"
	);
	

	/* ���������� GPU ���͵� CPU �� buffer */
	ck(cudaMemcpy(flow_buffer_cpu.get_pointer(), flow_buffer_gpu.get_pointer(), output_element_count * sizeof(float), cudaMemcpyDeviceToHost));


	/* ���� flow_buffer_cpu ������, չʾ */
	cv::Mat flow_visualize(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));
	flow_to_image_inplementation(
		flow_visualize.data,
		flow_buffer_cpu.get_pointer(),
		image_height,
		image_width
	);

	/* ���ӻ� */
	cv_show(flow_visualize);

	/* ���� */
	cv::imwrite("../images/output/sintel.png", flow_visualize, { cv::IMWRITE_PNG_COMPRESSION, 0});

	return 0;
}