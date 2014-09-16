#pragma once

#include "layer.h"
#include "util.h"

namespace mlp{
	class OutputLayer :public Layer
	{
	public:
		OutputLayer(size_t in_depth) :
			Layer(in_depth, 0)
		{}

		void forward(){
			this->err = 0;
			exp_y_vec.clear();
			//exp_y_vec.resize(in_depth_);
			exp_y_vec.push_back(this -> exp_y);
			//exp_y_vec[this->exp_y] = 1;
			for (size_t i = 0; i < in_depth_; i++){
				err += 0.5 * (exp_y_vec[i] - input_[i]) *
					(exp_y_vec[i] - input_[i]);
			}
			output_ = input_;
		}

		void back_prop(){
			/* compute err terms of output layers */
			g_.clear();

			for (size_t i = 0; i < in_depth_; i++){
				//std::cout << "df:" << df_sigmod(input_[i]) << std::endl;
				//std::cout << "exp_vec[i]: " << exp_y_vec[i] << std::endl;
				g_.push_back((exp_y_vec[i] - input_[i]) * df_sigmod(input_[i]));
			}
			//std::cout << "g_" << std::endl;
			//disp_vec_t(g_);
		}

		void init_weight(){}

	private:
	};
} // namespace mlp
