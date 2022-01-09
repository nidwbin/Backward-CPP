//
// Created by nidwbin on 2022/1/7.
//

#include "Tensor.h"
#include "Module.h"

#ifndef BACKWARD_LINEAR_H
#define BACKWARD_LINEAR_H


class Linear : protected Module {
private:
    Tensor weight;

    float lr;

    void update_weight(Tensor &);

protected:
    void solve_grid(Tensor &);

    void backward(Tensor &) override;

public:
    Linear(int in_size, int out_size, float learn_rate);

    void forward(Tensor &) override;

    void forward(Module *) override;

    Module *operator()(Module *) override;

    Module *operator()(Tensor &) override;
};


#endif //BACKWARD_LINEAR_H
