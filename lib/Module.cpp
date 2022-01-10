//
// Created by nidwbin on 2022/1/7.
//

#include "Module.h"

Module::Module(Tensor &input) {
    this->require_grid = false;
    this->input_module = nullptr;
    this->input_tensor = input;
    this->output = input;
}

Module::Module() = default;

void Module::set_require_grid(bool grid_flag) {
    this->require_grid = grid_flag;
}

Module *Module::operator()(Tensor &input) {
    this->require_grid = false;
    this->input_module = nullptr;
    this->input_tensor = input;
    this->output = input;
    this->forward(input);
    return this;
}

Module *Module::operator()(Module *input) {
    this->require_grid = true;
    this->input_module = input;
    this->forward(input);
    return this;
}

void Module::backward() {
    if (this->input_module) {
        if (this->input_module->require_grid) {
            this->input_module->backward(this->grid);
        } else {
            this->input_module->backward();
        }
    }
}


void Module::backward(Tensor &back_grid) {
    this->backward();
}

void Module::solve_grid() {

}

void Module::forward(Tensor &input) {
    this->output = input;
}

void Module::forward(Module *input) {
    this->output = input->output;
}