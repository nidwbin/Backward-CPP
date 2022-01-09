//
// Created by nidwbin on 2022/1/7.
//
#include "Tensor.h"
#include "Module.h"

#ifndef BACKWARD_SIGMOID_H
#define BACKWARD_SIGMOID_H


class Sigmoid : protected Module {
protected:
    void solve_grid(Tensor &);

    void backward(Tensor &) override;

public:
    Sigmoid();

    void forward(Module *) override;

    void forward(Tensor &) override;

    Module *operator()(Module *) override;

    Module *operator()(Tensor &) override;
};


#endif //BACKWARD_SIGMOID_H
