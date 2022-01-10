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
    std::vector<float> data;
    std::vector<int> _shape_;

    void mul_loop(Tensor &, Tensor &, int, std::vector<int> &, std::vector<int> &, std::vector<int> &);

public:

    Tensor(std::vector<int> &, float);

    explicit Tensor(std::vector<int> &);

    Tensor();

    int size() const;

    static Tensor zero_like(Tensor &);

    static Tensor ones_like(Tensor &);

    Tensor dot(Tensor &);

    Tensor dot(float);

    Tensor div(float);

    void update(std::vector<int> &, float);

    Tensor mul(Tensor &);

    void reshape(std::vector<int> &);

    std::vector<int> shape();

    float &operator[](int);

    float &operator()(int, ...);

    float &operator()(std::vector<int> &);

    Tensor operator+(Tensor);

    Tensor operator+(float);

    Tensor operator-(Tensor);

    Tensor operator-(float);

    Tensor operator*(Tensor);

    Tensor operator*(float);

    Tensor operator/(float);

};


#endif //BACKWARD_TENSOR_H
