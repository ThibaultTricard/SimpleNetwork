#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "mat.h"
#include <random>

typedef float input_t;

class input {
public:
	input() {};
	
	virtual vec<input_t> toVector() = 0;
	virtual vec<input_t> target() = 0;
};

input_t sig(input_t x) {
	return 1.0 / (1.0 + exp(-x));
}

class layer {
public :
	layer(int inputsize, int size) {
		m_inputWeight =  mat<input_t>(size, inputsize);
		m_iSize = inputsize;
		m_oSize = size;

		std::default_random_engine generator;
		std::normal_distribution<float> distribution(0.0, 1.0/sqrt(inputsize) /3.0);

		for (int i = 0; i < size; i++) {
			for (int j = 0; j < inputsize; j++) {
				m_inputWeight[i][j] = distribution(generator);
			}
		}
	};

	vec<input_t> evaluate( vec<input_t> input) {
		m_lastinput = vec<input_t>(input.size());
		for (int i = 0; i < input.size(); i++) {
			m_lastinput[i] = input[i];
		}
		m_lastOutput = m_inputWeight * input;
		for (int i = 0; i < m_lastOutput.size(); i++) {
			m_lastOutput[i] = sig(m_lastOutput[i]);
		}
		return m_lastOutput;
	}

	vec<input_t> correct(vec<input_t> error, input_t lrate) {
		mat<input_t> errWeight = errorMat();
		vec<input_t> prev_error = errWeight * error;


		mat<input_t> DeltaWeight(m_inputWeight.size(), m_inputWeight[0].size());
		for (int i = 0; i < m_inputWeight.size(); i++) {
			for (int j = 0; j < m_inputWeight[0].size(); j++) {
				DeltaWeight[i][j] = lrate * error[i] * m_lastOutput[i] * (1.0 - m_lastOutput[i]) * m_lastinput[j];
			}
		}
		m_inputWeight = m_inputWeight + DeltaWeight;
		return prev_error;
	}


	int m_iSize = 0;
	int m_oSize = 0;



private :

	mat<input_t> errorMat() {
		mat<input_t> error(m_inputWeight[0].size(), m_inputWeight.size());
		for (int i = 0; i < m_inputWeight.size(); i++) {
			for (int j = 0; j < m_inputWeight[0].size(); j++) {
				error[j][i] = m_inputWeight[i][j];
			}
		}
		return error;
	}

	mat<input_t> m_inputWeight = mat<input_t>(0,0);
	vec<input_t> m_lastOutput = vec<input_t>(0);
	vec<input_t> m_lastinput = vec<input_t>(0);
};

class network {
public :
	network(int inputSize, int outputSize, int nbLayer, int layerSize, float learningRate) {
		m_iSize = inputSize;
		m_oSize = outputSize;
		m_lRate = learningRate;
		m_nbLayer = nbLayer;
		m_layer = std::vector<layer*>(nbLayer + 2);
		m_layer[0] = new layer(inputSize, layerSize);
		for (int i = 1; i <= nbLayer ; i++) {
			m_layer[i] = new layer(layerSize, layerSize);
		}
		m_layer[nbLayer + 1] = new layer(layerSize, outputSize);
	}




	void train(std::vector<input*> inputs) {
		for (int i = 0; i < inputs.size(); i++) {
			vec<input_t>& input = inputs[i]->toVector();
			vec<input_t>& target = inputs[i]->target();
			if (input.size() == m_iSize && target.size() == m_oSize) {
				vec<input_t> res = evaluate(input);
				vec<input_t> error = target -res;
				
				//
				debug(i, res, target, error);

				for (int j = m_layer.size() - 1; j >= 0; j--) {
					error = m_layer[j]->correct(error, m_lRate);
				}
			}
		}
	}

	vec<input_t> evaluate( vec<input_t> input) {
		vec<input_t> out = input;
		for (int i = 0; i < m_layer.size(); i++) {
			out = m_layer[i]->evaluate(out);
		}
		return out;
	}


	void debug(int sample, vec<input_t> res, vec<input_t> target, vec<input_t> error) {
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


	


private : 
	int m_iSize;
	int m_oSize;
	input_t m_lRate;
	int m_nbLayer;
	std::vector<layer*> m_layer;
};