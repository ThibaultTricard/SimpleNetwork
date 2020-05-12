#include "mat.h"
#include "Network.h"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

class input1D : public input{
public:
	input1D(vec<input_t> data, vec<input_t> target) : input() {
		m_data = data;
		m_target = target;
	};

	virtual vec<input_t> toVector() override {
		return m_data;
	};
	virtual vec<input_t> target() override {
		return m_target;
	}

private : 
	vec<input_t> m_data;
	vec<input_t> m_target;
};

void main()
{
	
	int inputSize = 784;
	int nbCategory = 10;
	int nbLayer = 1;
	int layersize = 200;
	float learningRate = 0.1;
	int dataNb = 0;
	/*
	network* n = new network(inputSize, nbCategory, nbLayer, layersize, learningRate);

	
	//Training File
	ifstream trainingData("./mnist_train.csv", ios::in);
	std::vector<input*> inputs;
	
	if (trainingData)
	{
		string line;  
		while (getline(trainingData, line)) {
			cout << "reading data number " << dataNb << endl;
			std::stringstream ss(line);
			std::string item;
			vec<input_t> data = vec<input_t>(inputSize);
			vec<input_t> target = vec<input_t>(nbCategory);
			for (int i = 0; i < nbCategory; i++) {
				target[i] = 0.01f;
			}
			int i = 0;
			while (std::getline(ss, item, ','))
			{
				if (i == 0) {
					int index = std::stoi(item);
					target[index] = 0.99;
				}
				else {
					data[i - 1] = (float(std::stoi(item)) / 255.0f);
				}
				i++;
			}
			inputs.push_back(new input1D(data, target));

			dataNb++;

			//if (dataNb = 100000) break;
		}
	}

	cout << "training ..." << endl;
	n->train(inputs);
	
	n->save("test.net");
	*/
	network* n2 = new network("test.net");


	//test File
	ifstream testData("./mnist_test_10.csv", ios::in);
	std::vector<input*> test;
	if (testData)
	{
		string line;
		while (getline(testData, line)) {
			cout << "reading data number " << dataNb << endl;
			std::stringstream ss(line);
			std::string item;
			vec<input_t> data = vec<input_t>(inputSize);
			vec<input_t> target = vec<input_t>(nbCategory);
			for (int i = 0; i < nbCategory; i++) {
				target[i] = 0.01f;
			}
			int i = 0;
			while (std::getline(ss, item, ','))
			{
				if (i == 0) {
					int index = std::stoi(item);
					target[index] = 0.99;
				}
				else {
					data[i - 1] = (float(std::stoi(item)) / 255.0f);
				}
				i++;
			}
			test.push_back(new input1D(data, target));

			dataNb++;
		}
	}

	//test the training
	cout <<endl<<endl<< "test data: ";
	for (int i = 0; i < test.size(); i++) {
		vec<input_t> error = test[i]->target() - n2->evaluate(test[i]->toVector());
		cout << "err : " << i<< " : ";
		for (int j = 0; j < error.size(); j++) {
			cout << fixed << setprecision(4) << abs(error[j]) << " : ";
		}
		cout << endl;
	}
	
	return;
}