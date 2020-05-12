#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include "mat.h"
#include <random>
#include <fstream>

#define debugTraining 1

typedef float input_t;

class input {
public:
	input() {};
	
	virtual vec<input_t> toVector() = 0;
	virtual vec<input_t> target() = 0;
};

input_t sig(input_t x);

class layer {
public:
	layer(int inputsize, int size);

	layer(mat<input_t> weight) {
		m_inputWeight = weight;
	}

	vec<input_t> evaluate(vec<input_t> input);

	vec<input_t> propagateBack(vec<input_t>& error);

	void correct(vec<input_t>& error, input_t lrate);

	const mat<input_t> weight() { return m_inputWeight; };
	int inputSize() { return m_iSize; };
	int outputSize() { return m_oSize; };

private :

	int						m_iSize = 0;
	int						m_oSize = 0;
	mat<input_t>	m_inputWeight;
	vec<input_t>	m_lastOutput;
	vec<input_t>	m_lastinput;
};

class network {
public :
	network(int inputSize, int outputSize, int nbLayer, int layerSize, float learningRate);

	network(std::string path) {
		std::ifstream saveFile;
		saveFile = std::ifstream(path, std::ios::out | std::ios::binary);
		
		saveFile.read((char*)&m_iSize, sizeof(m_iSize));
		saveFile.read((char*)&m_oSize, sizeof(m_oSize));
		saveFile.read((char*)&m_lRate, sizeof(m_lRate));
		saveFile.read((char*)&m_nbLayer, sizeof(m_nbLayer));

		m_layer = std::vector<layer*>(m_nbLayer);

		for (int n = 0; n < m_nbLayer; n++) {
			int iSize;
			int oSize;
			saveFile.read((char*)&iSize, sizeof(iSize));
			saveFile.read((char*)&oSize, sizeof(oSize));
			mat<input_t> w (oSize, iSize);
			for (int i = 0; i < w.size(); i++) {
				saveFile.read((char*)&w[i][0], w[i].size() * sizeof(input_t));
			}
			m_layer[n] = new layer(w);
		}
		saveFile.close();
	}

	void train(std::vector<input*> inputs);

	vec<input_t> evaluate(vec<input_t>& input);

	void save(std::string path) {
		std::fstream saveFile;
		saveFile = std::fstream(path, std::ios::out | std::ios::binary);
		
		saveFile.write((char*)&m_iSize, sizeof(m_iSize));
		saveFile.write((char*)&m_oSize, sizeof(m_oSize));
		saveFile.write((char*)&m_lRate, sizeof(m_lRate));
		saveFile.write((char*)&m_nbLayer, sizeof(m_nbLayer));

		for (int n = 0; n < m_nbLayer; n++) {
			int iSize = m_layer[n]->inputSize();
			int oSize = m_layer[n]->outputSize();
			mat<input_t> w = m_layer[n]->weight();
			saveFile.write((char*)&iSize, sizeof(iSize));
			saveFile.write((char*)&oSize, sizeof(oSize));
			for (int i = 0; i < w.size(); i++) {
				saveFile.write((char*)&w[i][0], w[i].size() * sizeof(input_t));
			}
		}

		saveFile.close();
	}

	//used to evaluate the original error on the input
	void inputError(vec<input_t>& error);

#if debugTraining
	void debug(int sample, vec<input_t> res, vec<input_t> target, vec<input_t> error);
#endif

private : 
	int m_iSize;
	int m_oSize;
	input_t m_lRate;
	int m_nbLayer;
	std::vector<layer*> m_layer;
};