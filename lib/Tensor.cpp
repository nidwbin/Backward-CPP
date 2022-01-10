//
// Created by benjamin on 2021/12/31.
//

#include "cassert"
#include "stdexcept"
#include "Tensor.h"

Tensor::Tensor(std::vector<int> &shape, float value) {
    int length = 1;
    assert(shape.size() > 1);
    this->_shape_ = shape;
    for (auto &i: this->_shape_) {
        length *= i;
    }
    this->data = std::vector<float>(length, value);
}

Tensor::Tensor(std::vector<int> &shape) : Tensor(shape, 0) {}

Tensor::Tensor() = default;

int Tensor::size() const {
    return (int) data.size();
}

Tensor Tensor::zero_like(Tensor &source) {
    Tensor t(source._shape_);
    return t;
}

Tensor Tensor::ones_like(Tensor &source) {
    Tensor t(source._shape_, 1);
    return t;
}

Tensor Tensor::dot(Tensor &tensor) {
    assert(_shape_ == tensor._shape_);
    Tensor ret = zero_like(*this);
    for (int i = 0; i < data.size(); ++i) {
        ret[i] = data[i] * tensor[i];
    }
    return ret;
}

Tensor Tensor::dot(float number) {
    Tensor ret = zero_like(*this);
    for (int i = 0; i < data.size(); ++i) {
        ret[i] = data[i] * number;
    }
    return ret;
}

Tensor Tensor::div(float number) {
    if (number == 0) {
        throw std::domain_error("Div zero error!");
    }
    Tensor ret = zero_like(*this);
    for (int i = 0; i < data.size(); ++i) {
        ret[i] = data[i] / number;
    }
    return ret;
}

void Tensor::mul_loop(Tensor &source, Tensor &target, int dim, std::vector<int> &index1, std::vector<int> &index2,
                      std::vector<int> &index3) {
    if (dim == target._shape_.size() - 2) {
        for (int i = 0; i < target._shape_[dim]; ++i) {
            index3[index3.size() - 2] = i;
            index1[dim] = i;
            for (int j = 0; j < target._shape_[dim + 1]; ++j) {
                index3[index3.size() - 1] = j;
                index2[dim + 1] = j;
                for (int k = 0; k < _shape_[dim + 1]; ++k) {
                    index1[dim + 1] = index2[dim] = k;
                    target(index3) += this->operator()(index1) * source(index2);
                }
            }
        }
    } else {
        for (int i = 0; i < _shape_[dim]; ++i) {
            index1[dim] = i;
            index2[dim] = i;
            index3[dim] = i;
            mul_loop(source, target, dim + 1, index1, index2, index3);
        }
    }
}

Tensor Tensor::mul(Tensor &tensor) {
    assert(_shape_.size() == tensor._shape_.size());
    assert(_shape_[_shape_.size() - 1] == tensor._shape_[_shape_.size() - 2]);
    std::vector<int> new_shape(_shape_.size());
    for (int i = 0; i < _shape_.size(); ++i) {
        new_shape[i] = std::max(_shape_[i], tensor._shape_[i]);
    }
    new_shape[new_shape.size() - 2] = _shape_[_shape_.size() - 2];
    new_shape[new_shape.size() - 1] = tensor._shape_[_shape_.size() - 1];
    Tensor ret(new_shape);
    std::vector<int> index1(_shape_.size(), 0), index2(tensor._shape_.size(), 0), index3(ret._shape_.size(), 0);
    mul_loop(tensor, ret, 0, index1, index2, index3);
    return ret;
}

void Tensor::reshape(std::vector<int> &new_shape) {
    int new_length = 1;
    for (auto &i: new_shape) {
        new_length *= i;
    }
    assert(data.size() == new_length);
    _shape_ = new_shape;
}

std::vector<int> Tensor::shape() {
    return _shape_;
}

void Tensor::update(std::vector<int> &index, float value) {
    assert(index.size() == _shape_.size());
    int ind = index[0];
    for (int i = 1; i < _shape_.size(); ++i) {
        ind = ind * _shape_[i - 1] + index[i];
    }
    data[ind] = value;
}

float &Tensor::operator[](int index) {
    if (0 <= index && index < data.size()) {
        return data[index];
    } else {
        throw std::out_of_range("Tensor operator [] out of range!");
    }
}

int check_index(int index, int range) {
    if (index < range) {
        return index;
    } else {
        throw std::out_of_range("Tensor operator () out of range!");
    }
}

float &Tensor::operator()(int len, ...) {
    assert(len == _shape_.size());
    int index;
    va_list args;
    va_start(args, len);
    index = check_index(va_arg(args, int), _shape_[0]);
    for (int i = 1; i < len; ++i) {
        index = index * _shape_[i] + check_index(va_arg(args, int), _shape_[i]);
    }
    va_end(args);
    return this->operator[](index);
}

float &Tensor::operator()(std::vector<int> &index) {
    assert(index.size() == _shape_.size());
    int ind = check_index(index[0], _shape_[0]);
    for (int i = 1; i < _shape_.size(); ++i) {
        ind = ind * _shape_[i] + check_index(index[i], _shape_[i]);
    }
    return this->operator[](ind);
}

Tensor Tensor::operator+(Tensor tensor) {
    assert(_shape_ == tensor._shape_);
    Tensor ret = zero_like(*this);
    for (int i = 0; i < this->data.size(); ++i) {
        ret[i] = this->data[i] + tensor.data[i];
    }
    return ret;
}

Tensor Tensor::operator+(float number) {
    Tensor ret = zero_like(*this);
    for (int i = 0; i < this->data.size(); ++i) {
        ret[i] = this->data[i] + number;
    }
    return ret;
}

Tensor Tensor::operator-(Tensor tensor) {
    assert(_shape_ == tensor._shape_);
    Tensor ret = zero_like(*this);
    for (int i = 0; i < this->data.size(); ++i) {
        ret[i] = this->data[i] - tensor.data[i];
    }
    return ret;
}

Tensor Tensor::operator-(float number) {
    Tensor ret = zero_like(*this);
    for (int i = 0; i < this->data.size(); ++i) {
        ret[i] = this->data[i] - number;
    }
    return ret;
}

Tensor Tensor::operator*(Tensor tensor) {
    return dot(tensor);
}

Tensor Tensor::operator*(float number) {
    return dot(number);
}

Tensor Tensor::operator/(float number) {
    return div(number);
}