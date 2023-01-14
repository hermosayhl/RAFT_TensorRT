// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void cv_show(cv::Mat& image, const std::string& message = "crane") {
	cv::imshow(message, image);
	cv::waitKey(0);
	cv::destroyAllWindows();
}

bool cv_write(const std::string save_path, const cv::Mat& image) {
	return cv::imwrite(save_path, image, {cv::IMWRITE_PNG_COMPRESSION, 0});
}