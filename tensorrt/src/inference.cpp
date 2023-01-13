// C++
#include <string>
#include <assert.h>
#include <iostream>
#include <filesystem>
// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// CUDA
#include <cuda_runtime_api.h>
// TensorRT
#include <nvinfer.h>
// self
#include "visualize.h"




void cv_show(cv::Mat& image, const std::string& message = "crane") {
	cv::imshow(message, image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}


int main() {
	std::cout << "hello world!\n";

	std::cout << std::filesystem::current_path().string() << std::endl;

	auto image1 = cv::imread("../../pytorch/demo-frames/frame_0016.png");
	if (image1.empty()) {
		std::cout << "¶ÁÈ¡Ê§°Ü!\n";
		return -1;
	}
	cv_show(image1);
	return 0;
}