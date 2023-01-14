// C++
#include <string>
#include <fstream>
#include <iostream>
// TensorRT
#include <common.h>
#include <logger.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
using namespace nvinfer1;



/* 反序列化, 输入引擎路径, 返回一个引擎指针 */
ICudaEngine* deserialize(const std::string& tensorrt_engine_path) {
	/* 按照二进制读取 */
	std::ifstream engine_reader(tensorrt_engine_path, std::ios::in | std::ios::binary);
	if (!engine_reader) {
		std::cerr << "Cannot read " << tensorrt_engine_path << "\n";
		return nullptr;
	}
	/* 获取文件长度 */
	engine_reader.seekg(0, engine_reader.end);
	long int fsize = engine_reader.tellg();
	engine_reader.seekg(0, engine_reader.beg);
	if (fsize <= 0) {
		std::cerr << "Error! No content in " << tensorrt_engine_path << "\n";
		return nullptr;
	}
	/* 准备一个缓冲区, 接收文件中的内容 */
	std::vector<char> engine_string(fsize);
	engine_reader.read(engine_string.data(), fsize);
	if (engine_string.size() != fsize) {
		std::cerr << "Error! fsize != engine_string\n";
		return nullptr;
	}
	std::cout << "Succeeded reading the serilized engine\n";
	engine_reader.close();
	/* 将字符串中的内容解析成推理引擎 */
	initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
	IRuntime* runtime{ createInferRuntime(sample::gLogger) };
	ICudaEngine* engine = runtime->deserializeCudaEngine(engine_string.data(), fsize, nullptr);
	if (engine == nullptr) {
		std::cerr << "Failed to build the engine\n";
		return nullptr;
	}
	std::cout << "Succeeded building the engine\n";
	return engine;
}



/* 自定义 logger */
class Logger: public ILogger {
public:
	Severity reportable_severity;
	Logger(Severity severity=Severity::kINFO):
		reportable_severity(severity) {}
	void log(Severity severity, const char* msg) noexcept override {
		if (severity > reportable_severity) {
			return;
		}
		switch (severity) {
			case Severity::kINTERNAL_ERROR:
				std::cerr << "INTERNAL_ERROR==>";
				break;
			case Severity::kERROR:
				std::cerr << "ERROR==>";
				break;
			case Severity::kWARNING:
				std::cerr << "WARNING==>";
				break;
			case Severity::kINFO:
				std::cerr << "INFO==>";
				break;
			default:
				std::cerr << "UNKNOWN==>";
		}
		std::cerr << msg << std::endl;
	}
};
static Logger gLogger(ILogger::Severity::kERROR);
