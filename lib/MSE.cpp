//
// Created by nidwbin on 2022/1/7.
//

#include "cassert"
#include "MSE.h"

MSE::MSE() {
    this->require_grid = false;
}

void MSE::solve_grid() {
}

void MSE::forward(Module *predict, Tensor &target) {
    assert(target.shape() == predict->output.shape());
    std::vector<int> shape = {1, 1};
    this->input_module = predict;
    this->output = Tensor(shape);
    this->grid = Tensor::zero_like(target);
    for (int i = 0; i < target.size(); ++i) {
        this->grid[i] = target[i] - predict->output[i];
        this->output[0] += this->grid[i] * this->grid[i];
    }
    this->output[0] /= 2;
}

Module *MSE::operator()(Module *predict, Tensor &target) {
    forward(predict, target);
    return this;
}

float MSE::get_result() {
    return this->output[0];
}

void MSE::backward() {
    Module::backward();
}
