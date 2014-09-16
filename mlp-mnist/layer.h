#include "util.h"

#pragma once
namespace mlp{
	class Layer
	{
	public:
		Layer(size_t in_depth,
			size_t out_depth) :
			in_depth_(in_depth), out_depth_(out_depth)
		{}

		virtual void init_weight() = 0;
		virtual void forward() = 0;
		virtual void back_prop() = 0;

		float_t sigmod(float_t in){
			return 1.0 / (1.0 + std::exp(-in));
		}

		float_t df_sigmod(float_t f_x) {
			return f_x * (1.0 - f_x);
		}


		size_t in_depth_;
		size_t out_depth_;

		vec_t W_;
		vec_t b_;

		vec_t input_;
		vec_t output_;

		Layer* next;
		Layer* prev;

		float_t alpha_; // learning rate
		float_t lambda_; // wight decay
		vec_t g_; // err terms

		/*output*/
		float_t err;
		int exp_y;
		vec_t exp_y_vec;
	};
}