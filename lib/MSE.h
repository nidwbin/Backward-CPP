//
// Created by nidwbin on 2022/1/7.
//
#include "Module.h"

#ifndef BACKWARD_MSE_H
#define BACKWARD_MSE_H


class MSE : protected Module {
protected:
    void solve_grid() override;

public:
    MSE();

    void backward() override;

    void forward(Module *, Tensor &);

    Module *operator()(Module *, Tensor &);

    float get_result();
};


#endif //BACKWARD_MSE_H
