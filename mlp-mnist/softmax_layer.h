#include "util.h"

#pragma once

namespace mlp {
	/*
	Softmax Regression Classifer for mnist task.
	output == 10.
	*/
	class SoftmaxRegression
	{
	public:
		SoftmaxRegression(vec_t x) :
			in_size_(x.size()), out_size_(10), x_(x), alpha_(0.003), lambda_(0.1)
		{
			W_.resize(in_size_ * out_size_);
			b_.resize(out_size_);
		}

		vec_t forward(){
			for (size_t out = 0; out < out_size_; out++){
				output_[out] = dot(x_, get_W(out, in_size_, W_)) + b_[out];
			}
			return softmax(output_);
		}

	private:
		/*
		http://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
		--------------------------------
		Vector y = mlp(x); // output of the neural network without softmax activation function
		double ymax = maximal component of y
		for(int f = 0; f < y.rows(); f++)
		y(f) = exp(y(f) - ymax);
		y /= y.sum();
		--------------------------------
		*/

		vec_t softmax(vec_t &in){
			assert(in.size() > 0);
			float_t m = in[0];
			for (size_t i = 1; i < in.size(); i++)
				m = std::max(m, in[i]);

			for (auto &i : in)
				i = exp(i - m);

			float_t sum = std::accumulate(in.begin(), in.end(), 0);

			for (auto &i : in)
				i /= sum;

			return in;
		}

		size_t in_size_;
		size_t out_size_;

		float_t alpha_; // learning rate
		float_t lambda_; // weight decay
		vec_t x_;
		vec_t output_;
		vec_t W_;
		vec_t b_;
	};
}