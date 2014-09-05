#include "hidden_layer.h"
#include "mnist_parser.h"
#include "softmax_layer.h"
#include "network.h"
#include "util.h"

using namespace mlp;
using namespace std;

int main(){
	/*
	Mnist_Parser m;
	m.load_testing();
	//m.load_training();
	vec2d_t x;
	vec_t y;
	vec2d_t test_x;
	vec_t test_y;
	
	for (size_t i = 0; i < 60000; i++){
		x.push_back(m.train_sample[i]->image);
		y.push_back(m.train_sample[i]->label);
	}
	
	
	
	for (size_t i = 0; i < 10000; i++){
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}
	*/
	vec2d_t x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	vec_t y = { 0, 1, 1, 0 };
	
	Network n;

	n.add_layer(new HiddenLayer(2,2));
	n.add_layer(new HiddenLayer(2, 1));
	//n.add_layer(new SoftmaxRegression(10, 10));
	n.train(x, y, 4);
	//n.test(test_x, test_y);
	
	getchar();
	return 0;
}