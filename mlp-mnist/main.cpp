
#include "mnist_parser.h"
#include "fullyconnected_layer.h"
#include "network.h"
#include "util.h"

using namespace mlp;
using namespace std;

int main(){
	
	Mnist_Parser m;
	//m.load_testing();
	//m.load_training();
	vec2d_t x;
	vec_t y;
	/*
	for (size_t i = 0; i < 60000; i++){
		x.push_back(m.train_sample[i]->image);
		y.push_back(m.train_sample[i]->label);
	}
	
	
	
	for (size_t i = 0; i < 10000; i++){
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}
	*/
	vec2d_t test_x = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
	vec_t test_y = { 0, 1, 1, 0 };
	Mlp n;

	n.add_layer(new FullyConnectedLayer(2, 100));
	n.add_layer(new FullyConnectedLayer(100, 1));

	n.train(test_x, test_y, 4);
	n.test(test_x, test_y, 4);
	
	getchar();
	return 0;
}