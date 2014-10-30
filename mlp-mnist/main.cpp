#include "network.h"

using namespace mlp;
using namespace std;

int main(){
	vec2d_t train_x;
	vec_t train_y;
	vec2d_t test_x;
	vec_t test_y;

	/*加载MNIST数据集*/
	LOAD_MNIST_TEST(test_x, test_y);
	LOAD_MNIST_TRAIN(train_x, train_y);
	
	/*使用神经网络拟合XOR函数，快速验证神经网络正确性*/
	//vec2d_t XOR_x = { { 0, 0, 0, 0 }, { 0, 1, 1, 1 }, { 1, 0, 0, 0 }, { 1, 1, 1, 1 } };
	//vec_t XOR_y = { 0, 1, 1, 0 };
	
	Mlp n(0.03, 0.01);

	/*拟合XOR参数：*/
	//n.add_layer(new FullyConnectedLayer(4, 10, new sigmoid_activation));
	//n.add_layer(new FullyConnectedLayer(10, 1, new sigmoid_activation));

	//n.train(XOR_x, XOR_y, 4);

	/*MNIST拟合：*/
	n.add_layer(new FullyConnectedLayer(28 *28, 1000, new sigmoid_activation));
	n.add_layer(new FullyConnectedLayer(1000, 10, new sigmoid_activation));
	n.train(train_x, train_y, 60000);
	n.test(test_x, test_y, 10000);
	
	getchar();
	return 0;
}