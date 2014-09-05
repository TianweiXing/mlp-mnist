#include "util.h"

#pragma once

namespace mlp {
	/*
	The single hidden layer for mnist dataset task. 
	Using sigmod activation function
	
	*/
	class HiddenLayer :public Layer
	{
	public:
		HiddenLayer(size_t in_size, size_t out_size) :
			Layer(0.3, 0.001, in_size, out_size)
		{
			output_.resize(out_size_);
			W_.resize(out_size_ * in_size_);
			b_.resize(out_size);
			d_in_.resize(in_size_);
			g_.resize(in_size_);
			this->init_weight();
		}

		vec_t forward(){
			//std::cout << "hidden forward feeding" << std::endl;
			for (size_t out = 0; out < out_size_; out++){
				output_[out] = sigmod(dot(input_, get_W(out, in_size_, W_)) + b_[out]);
			}
			return output_;
		}

		void back_prop(){
			//std::cout << "hidden layer backprop" << std::endl;
			//assert(this->next != nullptr);
			//this -> calc_dinput();
			for (size_t in = 0; in < in_size_; in++){
				if (this->next == nullptr)
					g_[in] = df_sigmod(input_[in]) *dot(this->softmax_g, get_W_step(in));
				else
					g_[in] = df_sigmod(input_[in]) * dot(this->next->g_, get_W_step(in));
			}
			//disp_vec_t(output_);
			//update weight
			for (size_t out = 0; out < out_size_; out++){
				for (size_t in = 0; in < in_size_; in++){
					if (this->next == nullptr)
						W_[out * in_size_ + in] += alpha_* input_[in] * this->softmax_g[out];
					else
						W_[out * in_size_ + in] += alpha_* input_[in] * this->next->g_[out];
				}
				//disp_vec_t(W_);
				if (this->next == nullptr)
					b_[out] += this->softmax_g[out];
				else
					b_[out] += this ->next ->g_[out];
			}
			//disp_vec_t(W_);
		}

	private:
		/*
		for the activation sigmod,
		weight init as [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
		see also:http://deeplearning.net/tutorial/references.html#xavier10
		*/
		void init_weight(){
			/*
			uniform_rand(W_.begin(), W_.end(),
				-4 * 6 / std::sqrtf((float)(in_size_ + out_size_)),
				4 * 6 / std::sqrtf((float)(in_size_ + out_size_)));
			uniform_rand(b_.begin(), b_.end(),
				-4 * 6 / std::sqrtf((float)(in_size_ + out_size_)),
				4 * 6 / std::sqrtf((float)(in_size_ + out_size_)));
			*/
			uniform_rand(W_.begin(), W_.end(), -2, 2);
			uniform_rand(b_.begin(), b_.end(), -2, 2);
			
		}
	};
} //namespace mlp