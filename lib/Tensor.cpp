//
// Created by benjamin on 2021/12/31.
//

#include "cassert"
#include "stdexcept"
#include "Tensor.h"

Tensor::Tensor(std::vector<int> &shape, float value) {
    this->length = 1;
    assert(shape.size() > 1);
    this->shape = shape;
    for (auto &i: this->shape) {
        this->length *= i;
    }
    this->data = std::vector<float>(length, value);
}

Tensor::Tensor(std::vector<int> &shape) : Tensor(shape, 0) {}

Tensor::Tensor() {
    this->length = 0;
}

int Tensor::size() const {
    return this->length;
}

Tensor Tensor::zero_like(Tensor source) {
    Tensor t(source.shape);
    return t;
}

Tensor Tensor::ones_like(Tensor source) {
    Tensor t(source.shape, 1);
    return t;
}

Tensor &Tensor::dot(Tensor tensor) {
    assert(shape == tensor.shape);
    for (int i = 0; i < length; ++i) {
        data[i] *= tensor[i];
    }
    return *this;
}

Tensor &Tensor::dot(float number) {
    for (auto &i: data) {
        i *= number;
    }
    return *this;
}

Tensor &Tensor::div(float number) {
    if (number == 0) {
        throw std::domain_error("Div zero error!");
    }
    for (auto &i: data) {
        i /= number;
    }
    return *this;
}

void Tensor::mul_loop(Tensor &source, Tensor &target, int dim, std::vector<int> &index1, std::vector<int> &index2,
                      std::vector<int> &index3) {
    if (dim == target.shape.size() - 2) {
        for (int i = 0; i < target.shape[dim]; ++i) {
            index3[index3.size() - 2] = i;
            index1[dim] = i;
            for (int j = 0; j < target.shape[dim + 1]; ++j) {
                index3[index3.size() - 1] = j;
                index2[dim + 1] = j;
                for (int k = 0; k < shape[dim + 1]; ++k) {
                    index1[dim + 1] = index2[dim] = k;
                    target(index3) += this->operator()(index1) * source(index2);
                }
            }
        }
    } else {
        for (int i = 0; i < shape[dim]; ++i) {
            index1[dim] = i;
            index2[dim] = i;
            index3[dim] = i;
            mul_loop(source, target, dim + 1, index1, index2, index3);
        }
    }
}

Tensor Tensor::mul(Tensor &tensor) {
    assert(shape.size() == tensor.shape.size());
    assert(shape[shape.size() - 1] == tensor.shape[shape.size() - 2]);
    std::vector<int> new_shape(shape.size());
    for (int i = 0; i < shape.size(); ++i) {
        new_shape[i] = std::max(shape[i], tensor.shape[i]);
    }
    new_shape[new_shape.size() - 2] = shape[shape.size() - 2];
    new_shape[new_shape.size() - 1] = tensor.shape[shape.size() - 1];
    Tensor ret(new_shape);
    std::vector<int> index1(shape.size(), 0), index2(tensor.shape.size(), 0), index3(ret.shape.size(), 0);
    mul_loop(tensor, ret, 0, index1, index2, index3);
    return ret;
}

void Tensor::reshape(std::vector<int> &new_shape) {
    int new_length = 1;
    for (auto &i: new_shape) {
        new_length *= i;
    }
    assert(length == new_length);
    shape = new_shape;
}

void Tensor::update(std::vector<int> &index, float value) {
    assert(index.size() == shape.size());
    int ind = index[0];
    for (int i = 1; i < shape.size(); ++i) {
        ind = ind * shape[i - 1] + index[i];
    }
    data[ind] = value;
}

float &Tensor::operator[](int index) {
    if (0 <= index && index < length) {
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
    assert(len == shape.size());
    int index;
    va_list args;
    va_start(args, len);
    index = check_index(va_arg(args, int), shape[0]);
    for (int i = 1; i < len; ++i) {
        index = index * shape[i] + check_index(va_arg(args, int), shape[i]);
    }
    va_end(args);
    return this->operator[](index);
}

float &Tensor::operator()(std::vector<int> &index) {
    assert(index.size() == shape.size());
    int ind = check_index(index[0], shape[0]);
    for (int i = 1; i < shape.size(); ++i) {
        ind = ind * shape[i] + check_index(index[i], shape[i]);
    }
    return this->operator[](ind);
}

Tensor &Tensor::operator+(Tensor &tensor) {
    assert(this->length == tensor.length);
    for (int i = 0; i < this->length; ++i) {
        this->data[i] += tensor.data[i];
    }
    return *this;
}

Tensor &Tensor::operator+(float number) {
    for (int i = 0; i < this->length; ++i) {
        this->data[i] += number;
    }
    return *this;
}

Tensor &Tensor::operator-(Tensor &tensor) {
    assert(this->length == tensor.length);
    for (int i = 0; i < this->length; ++i) {
        this->data[i] -= tensor.data[i];
    }
    return *this;
}

Tensor &Tensor::operator-(float number) {
    for (int i = 0; i < this->length; ++i) {
        this->data[i] -= number;
    }
    return *this;
}

Tensor &Tensor::operator*(Tensor &tensor) {
    return dot(tensor);
}

Tensor &Tensor::operator*(float number) {
    return dot(number);
}

Tensor &Tensor::operator/(float number) {
    return div(number);
}