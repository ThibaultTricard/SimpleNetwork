#pragma once
#include <vector>

template<typename T>
class mat {
public :
	mat<T>(int row, int collumn) {
		m_value = std::vector<std::vector<T>>(row);
		for (int i = 0; i < row; i++) {
			m_value[i] = std::vector<T>(collumn);
		}
	};

	unsigned size() const{
		return m_value.size();
	};

	std::vector<T>& operator[] (unsigned i) {
		return this->m_value[i];
	};



private : 
	std::vector<std::vector<T>> m_value;
};

template<typename T>
class vec {
public :

	vec<T>() {

	};

	vec<T> (int row) {
		m_value = std::vector<T>(row);
	};

	unsigned size() const{
		return m_value.size();
	};

	T& operator[] (unsigned i) {
		return this->m_value[i];
	};

private:
	std::vector<T> m_value;
};

template<typename T>
mat<T> transpose(mat<T>& m) {
	mat<T> res(m[0].size(), m.size());
	for (int i = 0; i < m.size(); i++) {
		for (int j = 0; j < m[0].size(); j++) {
			res[j][i] = m[i][j];
		}
	}
	return res;
}

template<typename T>
mat<T> operator+(mat<T>& m1, mat<T>& m2) {
	mat<T> res(m1.size(), m1[0].size());
	for (int i = 0; i < m1.size(); i++) {
		for (int j = 0; j < m1[0].size(); j++) {
			res[i][j] = m1[i][j] + m2[i][j];
		}
	}
	return res;
}


template<typename T>
mat<T> operator*(mat<T>& m1, mat<T>& m2) {
	mat<T> res(m1.size(), m2[0].size());
	if (m1[0].size() == m2.size()) {
		for (int i = 0; i < m1.size(); i++) {
			for (int j = 0; j < m2[0].size(); j++) {
				for (int k = 0; k < m1[0].size(); k++) {
					res[i][j] += m1[i][k] * m2[k][j];
				}
			}
		}
	}
	return res;
}

template<typename T>
vec<T> operator*(mat<T>& m1, vec<T>& v) {
	vec<T> res(m1.size());
	if (m1[0].size() == v.size()) {
		for (int i = 0; i < m1.size(); i++) {
			for (int k = 0; k < m1[0].size(); k++) {
				res[i] += m1[i][k] * v[k];
			}
		}
	}
	return res;
}

template<typename T>
vec<T> operator-(vec<T>& v1, vec<T>& v2) {
	vec<T> res(v1.size());
	for (int i = 0; i < v1.size(); i++) {
		res[i] += v1[i] - v2[i];
	}
	return res;
}

template<typename T>
vec<T> operator*(vec<T>& v1, vec<T>& v2) {
	vec<T> res(v1.size());
	for (int i = 0; i < v1.size(); i++) {
		res[i] += v1[i] * v2[i];
	}
	return res;
}