//
// Created by benjamin on 2021/12/31.
//
#include "utility"
#include "vector"
#include "cstdarg"

#ifndef BACKWARD_TENSOR_H
#define BACKWARD_TENSOR_H


class Tensor {
private:
    int length;
    std::vector<float> data;

    void mul_loop(Tensor &, Tensor &, int, std::vector<int> &, std::vector<int> &, std::vector<int> &);

public:
    std::vector<int> shape;

    Tensor(std::vector<int> &, float);

    explicit Tensor(std::vector<int> &);

    Tensor();

    static Tensor zero_like(Tensor);

    static Tensor ones_like(Tensor);

    Tensor &dot(Tensor);

    Tensor &dot(float);

    Tensor &div(float);

    void update(std::vector<int> &, float);

    Tensor mul(Tensor &);

    void reshape(std::vector<int> &);

    float &operator[](int);

    float &operator()(int, ...);

    float &operator()(std::vector<int> &);

    Tensor &operator*(Tensor &);

    Tensor &operator*(float);

    Tensor &operator/(float);

};


#endif //BACKWARD_TENSOR_H
