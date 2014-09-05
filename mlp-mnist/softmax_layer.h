#include "util.h"

#pragma once

namespace mlp {
	/*
	Softmax Regression Classifer for mnist task.
	output == 10.
	last layer.
	weight decay.
	*/
	class SoftmaxRegression :public Layer
	{
	public:
		SoftmaxRegression(size_t in_size, size_t out_size) :
			Layer(0.03, 0.001, in_size, out_size)
		{
			output_.resize(out_size_);
			g_.resize(in_size_);
			d_in_.resize(in_size_);
			W_.resize(in_size_ * out_size_);
			b_.resize(out_size_);
		}

		vec_t forward(){
			//std::cout << "softmax forward feeding" << std::endl;
			for (size_t out = 0; out < out_size_; out++){
				output_[out] = dot(input_, get_W(out, in_size_, W_)) + b_[out];
			}
			//disp_vec_t(input_);
			return this->softmax(output_);
		}

		void back_prop(){
			//std::cout << "softmax backprop" << std::endl;
			this->calc_dinput();
			for (size_t in = 0; in < in_size_; in++){
				//g_[in] = d_in_[in] * dot(this->softmax_g, get_W_step(in));
				g_[in] = this->softmax_g[in];
			}
			
			vec_t g;
			for (size_t i = 0; i < out_size_; i++){
				float_t _ = 0 - output_[i];
				if (abs(softmax_exp_y[i] - 1.0) < 1e-7)
					_ = 1.0 - output_[i];
				g.push_back(_);
			}

			for (size_t out = 0; out < out_size_; out++){
				for (size_t in = 0; in < in_size_; in++){
					/*fuck*/
					W_[out * in_size_ + in] += alpha_ * (g[out] * input_[in]
						+ /*weight decay*/lambda_ * W_[out * in_size_ + in]);
				}
				b_[out] += alpha_ * g[out];
			}
			
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
	};
}