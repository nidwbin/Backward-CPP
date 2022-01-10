//
// Created by nidwbin on 2022/1/7.
//
#include <cassert>
#include "cmath"
#include "Sigmoid.h"

Sigmoid::Sigmoid() {
    this->require_grid = true;
}

void Sigmoid::solve_grid(Tensor &back_grid) {
    this->grid = this->output * (Tensor::ones_like(this->output) - this->output) * back_grid;
}

void Sigmoid::backward(Tensor &back_grid) {
    this->solve_grid(back_grid);
    this->Module::backward();
}

void Sigmoid::forward(Module *input) {
    this->input_module = input;
    this->output = Tensor::zero_like(input->output);
    for (int i = 0; i < this->output.size(); ++i) {
        this->output[i] = 1 / (1 + std::exp(-input->output[i]));
    }
}

void Sigmoid::forward(Tensor &input) {
    this->input_tensor = input;
    this->output = Tensor::zero_like(input);
    for (int i = 0; i < this->output.size(); ++i) {
        this->output[i] = 1 / (1 + std::exp(input[i]));
    }
}

Module *Sigmoid::operator()(Module *input) {
    forward(input);
    return this;
}

Module *Sigmoid::operator()(Tensor &input) {
    forward(input);
    return this;
}
