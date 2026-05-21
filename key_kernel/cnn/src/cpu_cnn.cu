#include "../include/cnn.cuh"
#include "../include/common.cuh"

void cpu_cnn(float* X, float* Y, float* W, int N, int M, int C, int H_in, int W_in, int H_out,
             int W_out, int K_H, int K_W) {
    for (int n = 0; n < N; ++n) {
        for (size_t m = 0; m < M; m++) {
            for (size_t ho = 0; ho < H_out; ho++) {
                for (size_t wo = 0; wo < W_out; wo++) {
                    Y[((n * M + m) * H_out + ho) * W_out + wo] = 0.0f;
                    for (size_t c = 0; c < C; c++) {
                        for (size_t kh = 0; kh < K_H; kh++) {
                            for (size_t kw = 0; kw < K_W; kw++) {
                                int h_in = ho + kh;
                                int w_in = wo + kw;
                                Y[((n * M + m) * H_out + ho) * W_out + wo] +=
                                    X[((n * C + c) * H_in + h_in) * W_in + w_in] *
                                    W[((m * C + c) * K_H + kh) * K_W + kw];
                            }
                        }
                    }
                }
            }
        }
    }
}