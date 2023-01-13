#include "visualize.h"


namespace {
	template<typename T>
	inline T square(const T x) {
		return x * x;
	}


	void get_color_wheel(std::vector<float>& color_wheel) {
		constexpr int RY = 15;
		constexpr int YG = 6;
		constexpr int GC = 4;
		constexpr int CB = 11;
		constexpr int BM = 13;
		constexpr int MR = 6;
		constexpr int ncols = RY + YG + GC + CB + BM + MR;
		color_wheel.resize(ncols * 3, 0.f);
		int col{0};
		for (int i = col; i < col + RY; ++i) {
			color_wheel[3 * i]     = 255;
			color_wheel[3 * i + 1] = std::floor(255 * ((i - col) / float(RY)));
		}
		col += RY;
		for (int i = col; i < col + YG; ++i) {
			color_wheel[3 * i]     = 255 - std::floor(255 * ((i - col) / float(YG)));
			color_wheel[3 * i + 1] = 255;
		}
		col += YG;
		for (int i = col; i < col + GC; ++i) {
			color_wheel[3 * i + 1] = 255;
			color_wheel[3 * i + 2] = std::floor(255 * ((i - col) / float(GC)));
		}
		col += GC;
		for (int i = col; i < col + CB; ++i) {
			color_wheel[3 * i + 1] = 255 - std::floor(255 * ((i - col) / float(CB)));
			color_wheel[3 * i + 2] = 255;
		}
		col += CB;
		for (int i = col; i < col + BM; ++i) {
			color_wheel[3 * i + 2] = 255;
			color_wheel[3 * i]     = std::floor(255 * ((i - col) / float(BM)));
		}
		col += BM;
		for (int i = col; i < col + MR; ++i) {
			color_wheel[3 * i + 2] = 255 - std::floor(255 * ((i - col) / float(MR)));
			color_wheel[3 * i]     = 255;
		}
	}
}

	


// 如果是视频推理, 或者是多帧推理, 这里的 vector 都可以拿到外面, 以免每次都申请空间释放空间
void flow_to_image_inplementation(
		unsigned char* result,
		float* flow,
		const int height,
		const int width) {
	// 准备一些临时变量
	const int length = height * width;
	std::vector<float> temp(length, 0.f);
	std::vector<float> temp_flow(length * 2, 0.f);
	// 找到 u, v
	float *u = flow, *v = flow + length;
	// 【1】 做第一步
	for (int i = 0; i < length; ++i) {
		temp[i] = std::sqrt(square(u[i]) + square(v[i]));
	}
	const float rad_max = *std::max_element(temp.begin(), temp.end());
	float *u_ = temp_flow.data();
	float *v_ = temp_flow.data() + length;
	for (int i = 0; i < length; ++i) {
		u_[i] = u[i] / (rad_max + 0.00001f);
		v_[i] = v[i] / (rad_max + 0.00001f);
	}
	// 【2】 准备一个 color_wheel
	std::vector<float> color_wheel;
	get_color_wheel(color_wheel);
	const int ncols = color_wheel.size() / 3;
	// 【3】 开始绘画
	float* rad = temp.data();
	for (int i = 0; i < length; ++i) {
		rad[i] = std::sqrt(square(u_[i]) + square(v_[i]));
	}
	std::vector<float> fk(length, 0.f);
	constexpr float PI{3.14159265358979323846};
	for (int i = 0; i < length; ++i) {
		float a = std::atan2(-v_[i], -u_[i]) / PI;
		fk[i] = (a + 1) / 2 * (ncols - 1);
	}
	std::vector<int> k0(length, 0);
	std::vector<int> k1(length, 0);
	for (int i = 0; i < length; ++i) {
		k0[i] = std::floor(fk[i]);
		k1[i] = k0[i] + 1;
		if (k1[i] == ncols) k1[i] = 0;
	}
	float *f = fk.data();
	for (int i = 0; i < length; ++i) {
		f[i] = fk[i] - k0[i];
	}
	for (int i = 0; i < length; ++i) {
		const int res_pos = 3 * i + 2;
		for (int c = 0; c < 3; ++c) {
			float col0 = color_wheel[k0[i] * 3 + c] / 255.f;
			float col1 = color_wheel[k1[i] * 3 + c] / 255.f;
			float col = (1.f - f[i]) * col0 + f[i] * col1;
			col = rad[i] <= 1.f ? 1.f - rad[i] * (1.f - col): col * 0.75f;
			result[res_pos - c] = std::floor(255 * col);
		}
	}
}


extern "C" {
	void flow_to_image(
			unsigned char* result,
			float* flow,
			const int height,
			const int width) {
		flow_to_image_inplementation(result, flow, height, width);
	}
}