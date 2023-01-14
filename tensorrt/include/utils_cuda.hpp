// C++
#include <iostream>
// CUDA
#include <cuda_runtime_api.h>


/* 检查 cuda 操作是否成功 */
inline bool check(cudaError_t e, int line_pos, const char* file_pos) {
	if (e != cudaSuccess) {
		std::cout << "CUDA Runtime Errot " << cudaGetErrorName(e) << " at line " << line_pos << "  in file " << file_pos << "\n";
		return false;
	}
	return true;
}
#define ck(call) check(call, __LINE__, __FILE__)


enum class CUDA_DATA_TYPE {
	NORMAL_MEMORY,   /* GPU 内存 */
	HOST_MEMORY      /* CPU 内存 */
};


/* CUDA 内存的智能指针 */
template<typename T>
class AutoCudaMallocPointer {
private:
	T* pointer          = nullptr;
	CUDA_DATA_TYPE sign = CUDA_DATA_TYPE::NORMAL_MEMORY;
public:
	int32_t storage     = 0;
	int32_t tensor_size = 0;
	std::string name    = "";
public:
	/* 构造函数和析构函数 */
	AutoCudaMallocPointer() {}
	AutoCudaMallocPointer(
		const int32_t tensor_size, 
		const CUDA_DATA_TYPE type=CUDA_DATA_TYPE::NORMAL_MEMORY, 
		const char* name="") {
		this->allocate(tensor_size, type, name);
	}
	~AutoCudaMallocPointer() noexcept {
		this->release();
	}
	/* 获取指针 */
	inline T* get_pointer() {
		return this->pointer;
	}

	/* 分配长度为 tensor_size 、类型为 type 的内存 */
	bool allocate(
		const int32_t tensor_size, 
		const CUDA_DATA_TYPE type=CUDA_DATA_TYPE::NORMAL_MEMORY, 
		const char* name="") {
		/* 如果之前 pointer 申请过内存, 释放它 */
		this->release();
		/* 设置新类型内存 */
		this->sign        = type;
		this->storage     = sizeof(T) * tensor_size;
		this->tensor_size = tensor_size;
		this->name        = name;
		switch (this->sign) {
			case CUDA_DATA_TYPE::NORMAL_MEMORY:
				ck(cudaMalloc((void**)&(this->pointer), this->storage));
				break;
			case CUDA_DATA_TYPE::HOST_MEMORY:
				ck(cudaMallocHost((void**)&(this->pointer), this->storage));
				break;
			default:
				std::cout << "Error: No such type of memory" << this->name << " can be allocated\n";
				return false;
		}
		if (this->pointer == nullptr) {
			std::cout << "Error: Memory  " << this->name << " cannot be allocated\n";
			return false;
		}
		return true;
	}
	/* 释放内存 */
	bool release() {
		if (this->pointer != nullptr) {
			switch (this->sign) {
				case CUDA_DATA_TYPE::NORMAL_MEMORY:
					ck(cudaFree(this->pointer));
					break;
				case CUDA_DATA_TYPE::HOST_MEMORY:
					ck(cudaFreeHost(this->pointer));
					break;
				default:
					std::cout << "Error: Memory" << this->name << " cannot be released\n";
					return false;
			}
			this->pointer = nullptr;
			std::cout << this->name << " is released\n";
			return true;
		}
		return false;
	}
};