#include "cstdio"
#include "lib/Tensor.h"

int main() {
    std::vector<int> shape1 = {1, 2}, shape2 = {2, 1};
    Tensor a(shape1, 1), b(shape2, 1), c;
    c = a.mul(b);
    for (auto i: c.shape) {
        printf("%d\n", i);
    }
    for (int i = 0; i < 1; ++i) {
        printf("\n");
        for (int j = 0; j < 1; ++j) {
            //for (int k = 0; k < 4; ++k) {
            printf("%f ", c(2, i, j));
            //}
        }
    }
    return 0;
}
