#include "hidden_layer.h"
#include "mnist_parser.h"
#include "softmax_layer.h"
#include "network.h"
#include "util.h"

using namespace mlp;
using namespace std;

int main(){
	
	Mnist_Parser m;
	m.load_testing();
	m.load_training();
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

	Network n;

	n.add_layer(new HiddenLayer(1024, 500));
	//n.add_layer(new HiddenLayer(800, 500));
	n.add_layer(new SoftmaxRegression(500, 10));
	n.train(x, y, 60000);
	n.test(test_x, test_y);

	getchar();
	return 0;
}