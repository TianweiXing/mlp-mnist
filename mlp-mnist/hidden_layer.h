#include "util.h"

#pragma once

namespace mlp {
	/*
	The single hidden layer for mnist dataset task. 
	Using sigmod activation function
	
	*/
	class HiddenLayer
	{
	public:
		HiddenLayer(vec_t input, size_t out_size) :
			input_(input), in_size_(input.size()), out_size_(out_size)
		{
			output_.resize(out_size_);
			W_.resize(out_size_ * in_size_);
			b_.resize(out_size);
			this->init_weight();
		}

		vec_t forward_prop(){
			for (size_t out = 0; out < out_size_; out++){
				output_[out] = sigmod(dot(input_, get_W(out, in_size_, W_)) + b_[out]);
			}
			return output_;
		}

	private:
		/*
		for the activation sigmod,
		weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
		see also:http://deeplearning.net/tutorial/references.html#xavier10
		*/
		void init_weight(){
			uniform_rand(W_.begin(), W_.end(),
				-4 * 6 / std::sqrtf((float)(in_size_ + out_size_)),
				4 * 6 / std::sqrtf((float)(in_size_ + out_size_)));
			uniform_rand(b_.begin(), b_.end(),
				-4 * 6 / std::sqrtf((float)(in_size_ + out_size_)),
				4 * 6 / std::sqrtf((float)(in_size_ + out_size_)));
		}

		float_t sigmod(float_t in){
			return 1.0 / (1.0 + std::exp(-in));
		}

		float_t df_sigmod(float_t f_x) {
			return f_x * (1.0 - f_x);
		}



		size_t in_size_;
		size_t out_size_;

		vec_t input_;
		vec_t output_;
		vec_t W_;
		vec_t b_;
	};
} //namespace mlp