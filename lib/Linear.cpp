//
// Created by nidwbin on 2022/1/7.
//

#include "vector"
#include "random"
#include "Linear.h"

Linear::Linear(int in_size, int out_size, float learn_rate) {
    this->require_grid = true;
    this->lr = learn_rate;
    std::vector<int> shape = {in_size, out_size};
    this->weight = Tensor(shape);
    std::random_device rand;
    std::default_random_engine e{rand()};
    std::uniform_real_distribution<float> u(-0.05, 0.05);
    for (int i = 0; i < this->weight.size(); ++i) {
        this->weight[i] = u(e);
    }
}

void Linear::update_weight(Tensor &back_grid) {
    Tensor input = this->input_module ? this->input_module->output : this->input_tensor;
    for (int i = 0; i < this->weight.shape()[0]; ++i) {
        for (int j = 0; j < this->weight.shape()[1]; ++j) {
            this->weight(2, i, j) = this->weight(2, i, j) + this->lr * back_grid(2, 0, j) * input(2, 0, i);
        }
    }
}

void Linear::solve_grid(Tensor &back_grid) {
    this->grid = Tensor::zero_like(this->input_module ? this->input_module->output : this->input_tensor);
    for (int i = 0; i < this->weight.shape()[0]; ++i) {
        for (int j = 0; j < this->weight.shape()[1]; ++j) {
            this->grid[i] += this->weight(2, i, j) * back_grid(2, 0, j);
        }
    }
}

void Linear::backward(Tensor &back_grid) {
    this->solve_grid(back_grid);
    this->update_weight(back_grid);
    this->Module::backward();
}

void Linear::forward(Tensor &input) {
    this->input_tensor = input;
    this->output = input.mul(this->weight);
}

void Linear::forward(Module *input) {
    this->input_module = input;
    this->output = input->output.mul(this->weight);
}

Module *Linear::operator()(Module *input) {
    forward(input);
    return this;
}

Module *Linear::operator()(Tensor &input) {
    forward(input);
    return this;
}