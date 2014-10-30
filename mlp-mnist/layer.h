#ifndef LAYER_H_
#define LAYER_H_
#pragma once
#include "activation.h"

namespace mlp{
	class Layer
	{
	public:
		Layer(size_t in_depth,
			size_t out_depth, activation* a) :
			in_depth_(in_depth), out_depth_(out_depth), a_(a)
		{}

		virtual void init_weight() = 0;
		virtual void forward() = 0;
		virtual void back_prop() = 0;

		size_t in_depth_;
		size_t out_depth_;

		vec_t W_;
		vec_t deltaW_; //last iter weight change for momentum;

		vec_t b_;

		activation* a_;

		vec_t input_;
		vec_t output_;

		Layer* next;

		float_t alpha_; // learning rate
		float_t lambda_; // momentum
		vec_t g_; // err terms

		/*output*/
		float_t err;
		int exp_y;
		vec_t exp_y_vec;
	};
} //namspace mlp

#endif