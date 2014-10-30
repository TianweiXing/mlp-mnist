#ifndef UTIL_H_
#define UTIL_H_

#pragma once

#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>

#include "boost\random.hpp"

namespace mlp {
	typedef std::vector<float_t> vec_t;
	typedef std::vector<std::vector<float_t>> vec2d_t;

	inline int uniform_rand(int min, int max) {
		static boost::mt19937 gen(0);
		boost::uniform_smallint<> dst(min, max);
		return dst(gen);
	}

	template<typename T>
	inline T uniform_rand(T min, T max) {
		static boost::mt19937 gen(0);
		boost::uniform_real<T> dst(min, max);
		return dst(gen);
	}

	template<typename Iter>
	void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
		for (Iter it = begin; it != end; ++it)
			*it = uniform_rand(min, max);
	}

	void disp_vec_t(const vec_t& v){
		for (auto i : v)
			std::cout << i << "\t";
		std::cout << "\n";
	}

	void disp_vec2d_t(const vec2d_t& v){
		for (auto i : v){
			for (auto i_ : i)
				std::cout << i_ << "\t";
			std::cout << "\n";
		}
	}

	float_t dot(const vec_t& x, const vec_t& w){
		assert(x.size() == w.size());
		float_t sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			sum += x[i] * w[i];
		}
		return sum;
	}

	vec_t f_muti_vec(float_t x, const vec_t& v){
		vec_t r;
		for_each(v.begin(), v.end(), [&](float_t i){
			r.push_back(x * i);
		});
		return r;
	}

	vec_t get_W(size_t index, size_t in_size_, const vec_t& W_){
		vec_t v;
		for (int i = 0; i < in_size_; i++){
			v.push_back(W_[index * in_size_ + i]);
		}
		return v;
	}
} // namespace mlp

#endif //UTIL_H_