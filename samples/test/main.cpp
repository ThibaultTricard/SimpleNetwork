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
	vec<input_t> m_data = vec<input_t>(0);
	vec<input_t> m_target = vec<input_t>(0);
};

int main()
{
	ifstream fichier("./mnist_train.csv", ios::in);  // on ouvre en lecture
	std::vector<input*> inputs;
	int nbCat = 10;
	int dataNb = 0;

	if (fichier)  // si l'ouverture a fonctionné
	{
		string contenu;  // déclaration d'une chaîne qui contiendra la ligne lue
		while (getline(fichier, contenu)) {  // on met dans "contenu" la ligne
			cout << "reading data number " << dataNb << endl;
			std::stringstream ss(contenu);
			std::string item;
			vec<input_t> data = vec<input_t>(784);
			vec<input_t> target = vec<input_t>(nbCat);
			for (int i = 0; i < nbCat; i++) {
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
			//if (dataNb == 10000) break;
		}
	}

	cout << "training" << endl;

	network* n = new network(784, nbCat, 0, 200, 0.01);

	n->train(inputs);
	return 0;
}