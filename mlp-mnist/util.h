#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>
#include "boost\random.hpp"

#pragma once
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

	void disp_vec_t(vec_t v){
		for (auto i : v)
			std::cout << i << "\t";
		std::cout << "\n";
	}

	void disp_vec2d_t(vec2d_t v){
		for (auto i : v){
			for (auto i_ : i)
				std::cout << i_ << "\t";
			std::cout << "\n";
		}
	}

	float_t dot(vec_t x, vec_t w){
		assert(x.size() == w.size());
		float_t sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			sum += x[i] * w[i];
		}
		return sum;
	}

	vec_t f_muti_vec(float_t x, const vec_t v){
		vec_t r;
		for_each(v.begin(), v.end(), [&](float_t i){
			r.push_back(x * i);
		});
		return r;
	}

	vec_t get_W(size_t index, size_t in_size_, vec_t W_){
		vec_t v;
		for (int i = 0; i < in_size_; i++){
			v.push_back(W_[index * in_size_ + i]);
		}
		return v;
	}

	struct Image {
		std::vector< std::vector<std::float_t> > img;// a image is represented by a 2-dimension vector  
		size_t size; // width or height

		// construction
		Image(size_t size_, std::vector< std::vector<std::float_t> > img_) :img(img_), size(size_){}

		// display the image
		void display(){
			for (size_t i = 0; i < size; i++){
				for (size_t j = 0; j < size; j++){
					if (img[i][j] > 200)
						std::cout << 1;
					else
						std::cout << 0;
				}
				std::cout << std::endl;
			}
		}

		// up size to 32, make up with 0
		void upto_32(){
			assert(size < 32);

			std::vector<std::float_t> row(32, 0);

			for (size_t i = 0; i < size; i++){
				img[i].insert(img[i].begin(), 0);
				img[i].insert(img[i].begin(), 0);
				img[i].push_back(0);
				img[i].push_back(0);
			}
			img.insert(img.begin(), row);
			img.insert(img.begin(), row);
			img.push_back(row);
			img.push_back(row);

			size = 32;
		}

		std::vector<std::float_t> extend(){
			std::vector<float_t> v;
			for (size_t i = 0; i < size; i++){
				for (size_t j = 0; j < size; j++){
					v.push_back(img[i][j]);
				}
			}
			return v;
		}
	};

	typedef Image* Img;

	struct Sample
	{
		uint8_t label; // label for a specific digit
		std::vector<float_t> image;
		Sample(float_t label_, std::vector<float_t> image_) :label(label_), image(image_){}
	};

	struct Layer
	{
		Layer* prev;
		Layer* next;
		vec_t input_;
		vec_t output_;
		vec_t g_;
		float_t alpha_; // learning rate
		float_t lambda_; // weight decay

		size_t in_size_;
		size_t out_size_;
		vec_t d_in_;
		vec_t W_;
		vec_t b_;

		/*eh..*/
		vec_t softmax_g;
		vec_t softmax_exp_y;

		virtual vec_t forward() = 0;
		virtual void back_prop() = 0; 

		void calc_dinput(){
			for (size_t i = 0; i < in_size_; i++){
				d_in_[i] = df_sigmod(input_[i]);
			}
		}

		float_t sigmod(float_t in){
			return 1.0 / (1.0 + std::exp(-in));
		}

		float_t df_sigmod(float_t f_x) {
			return f_x * (1.0 - f_x);
		}

		vec_t get_W_step(size_t in){
			vec_t r;
			for (size_t i = in; i < out_size_ * in_size_; i += in_size_){
				r.push_back(W_[i]);
			}
			return r;
		}

		Layer(float_t alpha, float_t lambda, size_t in_size, size_t out_size) :
			alpha_(alpha), lambda_(lambda), in_size_(in_size), out_size_(out_size){}
	};
} // namespace mlp