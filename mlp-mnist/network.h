#include "util.h"

#pragma once
namespace mlp{
	class Network
	{
	public:
		void train(vec2d_t train_x, vec_t train_y, int train_size){
			train_x_ = train_x, train_y_ = train_y, train_size_ = train_size;
			for (size_t i = 0; i < 3000 * train_size_; i++){
				std::cout << "training loop:" << i << std::endl;
				train_once();
			}
		}

		void test(vec2d_t test_x, vec_t test_y){
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_x_.size();
			size_t bang = 0;
			for (size_t i = 0; i < test_x.size(); i++){
				std::cout << "testing loop:" << i << std::endl;
				if (test_once()){
					bang++;
				}
			}
			std::cout << (float)bang / 10000 << std::endl;
		}

		void add_layer(Layer* layer){
			if (!layers.empty())
				this->layers.back()->next = layer;
			this->layers.push_back(layer);
			layer->next = NULL;
		}

	private:
		size_t max_iter(vec_t v){
			size_t i = 0;
			float_t max = v[0];
			for (size_t j = 1; j < v.size(); j++){
				if (v[j] > max){
					max = v[j];
					i = j;
				}
			}
			return i;
		}

		bool test_once(){
			auto test_x_index = uniform_rand(0, test_size_ - 1);
			std::cout << "train y: " << train_y_[test_x_index] << std::endl;
			layers[0]->input_ = test_x_[test_x_index];
			for (auto layer : layers){
				layer->forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			//std::cout << "forward feeding over" << std::endl;
			//disp_vec_t(layers.back()->output_);
			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}

		void train_once(){
			auto train_x_index = uniform_rand(0, train_size_ - 1);
			std::cout << "train y: " << train_y_[train_x_index] << std::endl;
			layers[0]->input_ = train_x_[train_x_index];
			for (auto layer : layers){
				layer->forward();
				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}
			//std::cout << "forward feeding over" << std::endl;
			disp_vec_t(layers.back()->output_);
			//std::cout << max_iter(layers.back()->output_) << std::endl;
			vec_t exp_y(layers.back()->output_.size(), 0);
			//exp_y[(int)train_y_[train_x_index]] = 1.0;
			exp_y[0] = train_y_[train_x_index];
			float_t err = 0;
			for (size_t i = 0; i < layers.back()->output_.size(); i++){
				err += 0.5 * (exp_y[i] - layers.back()->output_[i]) * 
					(exp_y[i] - layers.back()->output_[i]);
			}
			std::cout << "err:" << err << std::endl;
			/*
			layers.back()->softmax_exp_y = exp_y;
			*/
			vec_t g_y;
			for (size_t i = 0; i < layers.back()->output_.size(); i++){
				g_y.push_back((exp_y[i] - layers.back()->output_[i]) * df_sigmod(layers.back()->output_[i]));
			}
			layers.back()->softmax_g = g_y;
			//disp_vec_t(g_y);
			for (auto i = layers.rbegin(); i != layers.rend(); i++){
				(*i)->back_prop();
			}
		}

		float_t df_sigmod(float_t f_x) {
			return f_x * (1.0 - f_x);
		}

		std::vector<Layer*> layers;
		
		vec2d_t train_x_;
		int train_size_;
		vec_t train_y_;

		vec2d_t test_x_;
		int test_size_;
		vec_t test_y_;
	};
}