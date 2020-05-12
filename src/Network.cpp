#include "Network.h"

input_t sig(input_t x){
	return input_t(1.0 / (1.0 + exp(-x)));
}


layer::layer(int inputsize, int size) {
	m_inputWeight = mat<input_t>(size, inputsize);
	m_iSize = inputsize;
	m_oSize = size;

	//Weight generation using a random normal law cetered arroud 0 beetween -1.0/sqrt(inputsize) and +1.0/sqrt(inputsize)
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, 1.0 / sqrt(inputsize) / 3.0);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < inputsize; j++) {
			m_inputWeight[i][j] = distribution(generator);
		}
	}
};

vec<input_t> layer::evaluate(vec<input_t> input) {
	//we keep a copy of the input and the output
	//for the weight correction
	m_lastinput = input;
	m_lastOutput = m_inputWeight * input;
	for (int i = 0; i < m_lastOutput.size(); i++) {
		m_lastOutput[i] = sig(m_lastOutput[i]);
	}
	return m_lastOutput;
}

vec<input_t> layer::propagateBack(vec<input_t>& error) {
	return transpose(m_inputWeight) * error;
}

void layer::correct(vec<input_t>& error, input_t lrate) {
	//we do a gradiate descente modulated by the learning rate on the weight
	for (int i = 0; i < m_inputWeight.size(); i++) {
		for (int j = 0; j < m_inputWeight[0].size(); j++) {
			m_inputWeight[i][j] += lrate * error[i] * m_lastOutput[i] * (1.0 - m_lastOutput[i]) * m_lastinput[j];
		}
	}
}


network::network(int inputSize, int outputSize, int nbLayer, int layerSize, float learningRate) {
	m_iSize = inputSize;
	m_oSize = outputSize;
	m_lRate = learningRate;
	m_nbLayer = nbLayer +1 ;
	m_layer = std::vector<layer*>(m_nbLayer);
	m_layer[0] = new layer(inputSize, layerSize);
	for (int i = 1; i < nbLayer; i++) {
		m_layer[i] = new layer(layerSize, layerSize);
	}
	m_layer[nbLayer] = new layer(layerSize, outputSize);
}

void network::train(std::vector<input*> inputs) {
	for (int i = 0; i < inputs.size(); i++) {
		vec<input_t>& input = inputs[i]->toVector();
		vec<input_t>& target = inputs[i]->target();
		if (input.size() == m_iSize && target.size() == m_oSize) {
			vec<input_t> res = evaluate(input);
			vec<input_t> error = target - res;

#if debugTraining
			debug(i, res, target, error);
#endif
			vec<input_t> prev_error;
			for (int j = m_layer.size() - 1; j >= 0; j--) {
				prev_error = m_layer[j]->propagateBack(error);
				m_layer[j]->correct(error, m_lRate);
				error = prev_error;
			}
		}
	}
}

vec<input_t> network::evaluate(vec<input_t>& input) {
	vec<input_t> out = input;
	for (int i = 0; i < m_layer.size(); i++) {
		out = m_layer[i]->evaluate(out);
	}
	return out;
}

//used to evaluate the original error on the input
void network::inputError(vec<input_t>& error) {
	for (int j = m_layer.size() - 1; j >= 0; j--) {
		error = m_layer[j]->propagateBack(error);
	}
}

#if debugTraining
void network::debug(int sample, vec<input_t> res, vec<input_t> target, vec<input_t> error) {
	//debug
	std::cout << "sample " << sample << std::endl;
	std::cout << "res : ";
	for (int j = 0; j < error.size(); j++) {
		std::cout << std::fixed << std::setprecision(4) << res[j] << " : ";
	}
	std::cout << std::endl;

	std::cout << "tar : ";
	for (int j = 0; j < error.size(); j++) {
		std::cout << std::fixed << std::setprecision(4) << target[j] << " : ";
	}
	std::cout << std::endl;

	std::cout << "err : ";
	for (int j = 0; j < error.size(); j++) {
		std::cout << std::fixed << std::setprecision(4) << abs(error[j]) << " : ";
	}
	std::cout << std::endl << std::endl;
}
#endif