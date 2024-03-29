#ifndef FULLYCONNECTEDLAYER_H_
#define FULLYCONNECTEDLAYER_H_

#pragma once
#include "layer.h"
#include "util.h"

namespace mlp{

	class FullyConnectedLayer :public Layer
	{
	public:
		FullyConnectedLayer(size_t in_depth, size_t out_depth, activation* a) :
			Layer(in_depth, out_depth, a)
		{
			output_.resize(out_depth_);
			W_.resize(in_depth_ * out_depth_);
			deltaW_.resize(in_depth_ * out_depth_);
			b_.resize(out_depth_);
			g_.resize(in_depth_);

			this->init_weight();
		}

		void forward(){
			for (size_t out = 0; out < out_depth_; out++){
				output_[out] = a_->f(dot(input_, get_W(out)) + b_[out]);
			}
		}

		void back_prop(){
			/*
			Compute the err terms;
			*/
			for (size_t in = 0; in < in_depth_; in++){
				g_[in] = a_->df(input_[in]) * dot(this->next->g_, get_W_step(in));
			}
			/*
			Update weights.
			*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					auto delta = alpha_/*learning rate*/
						* input_[in] * this->next->g_[out]/*err terms*/
						/*lambda_ momentum*/
						+ lambda_ * deltaW_[out * in_depth_ + in];
					W_[out * in_depth_ + in] += delta;
					/*update momentum*/
					deltaW_[out * in_depth_ + in] = delta;
				}
				b_[out] += this->next->g_[out];
			}
		}

		/*
		for the activation sigmod,
		weight init ： [-4 * (6 / sqrt(fan_in + fan_out)), +4 *(6 / sqrt(fan_in + fan_out))]:
		see also:http://deeplearning.net/tutorial/references.html#xavier10
		*/
		void init_weight(){
			/*
			uniform_rand(W_.begin(), W_.end(),
			-4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
			4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
			uniform_rand(b_.begin(), b_.end(),
			-4 * 6 / std::sqrtf((float)(fan_in() + fan_out())),
			4 * 6 / std::sqrtf((float)(fan_in() + fan_out())));
			*/
			uniform_rand(W_.begin(), W_.end(), -1, 1);
			uniform_rand(b_.begin(), b_.end(), -1, 1);

		}
	private:
		vec_t get_W(size_t index){
			vec_t v;
			for (int i = 0; i < in_depth_; i++){
				v.push_back(W_[index * in_depth_ + i]);
			}
			return v;
		}

		vec_t get_W_step(size_t in){
			vec_t r;
			for (size_t i = in; i < out_depth_ * in_depth_; i += in_depth_){
				r.push_back(W_[i]);
			}
			return r;
		}
	};
} // namespace mlp

#endif 