//
// Created by nidwbin on 2022/1/7.
//

#include "Tensor.h"

#ifndef BACKWARD_MODULE_H
#define BACKWARD_MODULE_H


class Module {
protected:
    bool require_grid{};

    Tensor grid;
    Tensor input_tensor;

    Module *input_module{};

    virtual void solve_grid();

public:
    Tensor output;

    explicit Module(Tensor &);

    Module();


    void set_require_grid(bool);

    virtual void backward();

    virtual void backward(Tensor &);

    virtual void forward(Tensor &);

    virtual void forward(Module *);

    virtual Module *operator()(Tensor &);

    virtual Module *operator()(Module *);
};


#endif //BACKWARD_MODULE_H
