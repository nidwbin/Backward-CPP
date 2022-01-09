#include "cassert"
#include "string"
#include "random"
#include "iostream"
#include "fstream"
#include "sstream"
#include "lib/Linear.h"
#include "lib/Tensor.h"
#include "lib/Sigmoid.h"
#include "lib/MSE.h"
#include "lib/Module.h"

#define data_rows 699
#define data_cols 11


bool load_data(Tensor &data, std::string &data_path) {
    std::vector<int> data_shape = {data_rows, data_cols};
    data = Tensor(data_shape);
    std::string line;
    int row, col;
    row = 0;
    printf("Load data ... ");
    try {
        std::ifstream file(data_path);
        assert(file.good());
        while (std::getline(file, line)) {
            std::string number;
            std::istringstream read(line);
            for (col = 0; col < data_cols; ++col) {
                std::getline(read, number, ',');
                data(2, row, col) = std::strtof(number.c_str(), nullptr);
            }
            ++row;
        }

    } catch (std::exception &exception) {
        std::cout << exception.what() << std::endl;
        return false;
    }
    printf("Done\n");
    return true;
}

bool copy_data(Tensor &source, Tensor &target, int index, int start, int end) {
    try {
        for (int i = start; i < end; ++i) {
            target(2, 0, i - start) = source(2, index, i);
        }
    } catch (std::exception &exception) {
        std::cout << exception.what() << std::endl;
        return false;
    }
    return true;
}

void generate_data(Tensor &data) {
    float total;
    std::random_device rand;
    std::default_random_engine e{rand()};
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<int> data_shape = {data_rows, data_cols};
    data = Tensor(data_shape);
    printf("Generate data ... ");
    for (int i = 0; i < data_rows; ++i) {
        total = 0;
        for (int j = 0; j < data_cols - 1; ++j) {
            data(2, i, j) = u(e);
            total += data(2, i, j);
        }
        data(2, i, data_cols - 1) = total > 0 ? 1 : 0;
    }
    printf("Done\n");
}

float get_predict(float predict) { return predict > 0.5 ? 1 : 0; }

void train(int run_mode, int start, int end_epoch, float end_loss, Tensor &data, Tensor &target, Tensor &data_to_input,
           Linear &hidden, Sigmoid &sigmoid_hidden, Linear &output, Sigmoid &sigmoid_output, MSE &loss,
           float train_rate = 0.8) {
    int epoch = 0;
    int train_number = data_rows * train_rate;
    float avg_loss;

    Module input, *pointer;
    printf("Training ...\n");
    do {
        avg_loss = 0;
        for (int i = 0; i < train_number; ++i) {
            copy_data(data, data_to_input, i, start, data_cols - 1);
            target[0] = data(2, i, data_cols - 1);
            target[0] = run_mode == 0 ? target[0] : target[0] == 4 ? 1 : 0;

            input = Module(data_to_input);
            pointer = hidden(&input);
            pointer = sigmoid_hidden(pointer);
            pointer = output(pointer);
            pointer = sigmoid_output(pointer);
            loss(pointer, target);
            avg_loss += loss.get_result();
            printf("\rEpoch: %d, Loss: %.8f", epoch + 1, avg_loss / (float) (i + 1));
            loss.backward();
        }
        printf("\n");
        ++epoch;
    } while (avg_loss > end_loss && epoch < end_epoch);
    printf("Done\n");
}

void test(int run_mode, int start, Tensor &data, Tensor &target, Tensor &data_to_input,
          Linear &hidden, Sigmoid &sigmoid_hidden, Linear &output, Sigmoid &sigmoid_output, MSE &loss,
          float train_rate = 0.8) {
    int test_number = 0;
    float predict;
    float right_number = 0;

    Module input, *pointer;
    printf("Testing ... \n");
    for (int i = data_rows * train_rate; i < data_rows; ++i) {
        ++test_number;
        copy_data(data, data_to_input, i, start, data_cols - 1);
        target[0] = data(2, i, data_cols - 1);
        target[0] = run_mode == 0 ? target[0] : target[0] == 4 ? 1 : 0;
        input = Module(data_to_input);
        pointer = hidden(&input);
        pointer = sigmoid_hidden(pointer);
        pointer = output(pointer);
        pointer = sigmoid_output(pointer);
        predict = get_predict(pointer->output[0]);
        if (predict == target[0]) {
            ++right_number;
        }
    }
    printf("Accuracy: %.4f%\n", 100 * std::min(right_number / (float) test_number, (float) 1));
    printf("Done\n");
}

void run(std::string &data_path, int run_mode = 0, int end_epoch = 10, float learn_rate = 0.1,
         int hidden_size = 32, int output_size = 1, float train_rate = 0.8, float end_loss = 1e-6) {
    int start = run_mode == 0 ? 0 : 1;
    int input_size = run_mode == 0 ? data_cols - 1 : data_cols - 2;

    std::vector<int> input_shape = {1, input_size};
    std::vector<int> target_shape = {1, 1};

    Tensor data, data_to_input(input_shape), target(target_shape);
    Linear hidden(input_size, hidden_size, learn_rate);
    Sigmoid sigmoid_hidden;
    Linear output(hidden_size, output_size, learn_rate);
    Sigmoid sigmoid_output;
    MSE loss;

    if (run_mode) {
        load_data(data, data_path);
    } else {
        generate_data(data);
    }
    train(run_mode, start, end_epoch, end_loss, data, target, data_to_input,
          hidden, sigmoid_hidden, output, sigmoid_output, loss,
          train_rate);
    test(run_mode, start, data, target, data_to_input,
         hidden, sigmoid_hidden, output, sigmoid_output, loss,
         train_rate);
}

int main(int argc, char **argv) {
    std::string data_path = "../data/breast-cancer-wisconsin.data";
    int run_mode = 0;
    if (argc >= 2) {
        run_mode = std::strtol(argv[1], nullptr, 10);
        switch (run_mode) {
            case 0:
            case 1:
                break;
            default: {
                printf("Error parameter. Only accept 0 or 1.\n");
                return 0;
            }
        }
    }
    if (argc == 3) {
        data_path = argv[2];
    }
    run(data_path, run_mode);
    return 0;
}
