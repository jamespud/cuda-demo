#include <cstdlib>
#include <exception>
#include <string>

#include "../include/cnn.cuh"
#include "../include/common.cuh"

// Default CNN parameters (NCHW layout, no padding, stride=1)
constexpr int kDefaultN = 4;
constexpr int kDefaultM = 16;
constexpr int kDefaultC = 3;
constexpr int kDefaultH_in = 224;
constexpr int kDefaultW_in = 224;
constexpr int kDefaultK_H = 3;
constexpr int kDefaultK_W = 3;

struct PerfStats {
    double time_ms;
    double gflops;
    bool passed;
    double speedup;
};

double compute_gflops(double ms, int N, int M, int C, int H_out, int W_out, int K_H, int K_W) {
    // 2 ops per MAC (multiply + accumulate)
    double ops = 2.0 * N * M * C * H_out * W_out * K_H * K_W;
    return ops / 1e9 / (ms / 1000.0);
}

double compute_speedup(double baseline_ms, double measured_ms) {
    return measured_ms > 0.0 ? baseline_ms / measured_ms : 0.0;
}

void init_data(std::vector<float>& data) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) {
        v = dist(rng);
    }
}

bool validate_output(const std::vector<float>& ref, const std::vector<float>& out,
                     float tol = 1e-3f) {
    if (ref.size() != out.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::fabs(ref[i] - out[i]) > tol) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] << " got=" << out[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

bool parse_positive_int(const char* text, int& value) {
    try {
        std::string token(text);
        size_t consumed = 0;
        long long parsed = std::stoll(token, &consumed);
        if (consumed != token.size() || parsed <= 0) return false;
        value = static_cast<int>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [N] [M] [C] [H_in] [W_in] [K_H] [K_W]\n"
              << "  All arguments are optional positional integers (must be positive).\n"
              << "  Defaults: N=" << kDefaultN << " M=" << kDefaultM << " C=" << kDefaultC
              << " H_in=" << kDefaultH_in << " W_in=" << kDefaultW_in << " K_H=" << kDefaultK_H
              << " K_W=" << kDefaultK_W << "\n";
}

void print_performance_table(const PerfStats& cpu, const PerfStats& native,
                             const PerfStats& flattened, const PerfStats& cudnn) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Version" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS" << std::setw(12) << "Speedup" << "Status" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "CPU" << std::setw(15) << std::fixed
              << std::setprecision(3) << cpu.time_ms << std::setw(15) << cpu.gflops << std::setw(12)
              << "1.00x" << "N/A" << std::endl;
    std::cout << std::left << std::setw(15) << "Native CUDA" << std::setw(15) << native.time_ms
              << std::setw(15) << native.gflops << std::setw(12) << std::fixed
              << std::setprecision(2) << native.speedup << "x" << (native.passed ? "PASS" : "FAIL")
              << std::endl;
    std::cout << std::left << std::setw(15) << "Flattened CUDA" << std::setw(15)
              << flattened.time_ms << std::setw(15) << flattened.gflops << std::setw(12)
              << std::fixed << std::setprecision(2) << flattened.speedup << "x"
              << (flattened.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << std::left << std::setw(15) << "cuDNN" << std::setw(15) << cudnn.time_ms
              << std::setw(15) << cudnn.gflops << std::setw(12) << std::fixed
              << std::setprecision(2) << cudnn.speedup << "x" << (cudnn.passed ? "PASS" : "FAIL")
              << std::endl;
}

int main(int argc, char** argv) {
    // 支持 -h/--help 打印 usage
    if (argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }
    if (!(argc == 1 || argc == 8)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int N = kDefaultN, M = kDefaultM, C = kDefaultC;
    int H_in = kDefaultH_in, W_in = kDefaultW_in;
    int K_H = kDefaultK_H, K_W = kDefaultK_W;

    if (argc == 8) {
        int* targets[] = {&N, &M, &C, &H_in, &W_in, &K_H, &K_W};
        const char* names[] = {"N", "M", "C", "H_in", "W_in", "K_H", "K_W"};
        for (int i = 0; i < 7; ++i) {
            if (!parse_positive_int(argv[i + 1], *targets[i])) {
                std::cerr << "Invalid argument for " << names[i]
                          << ": must be a positive integer.\n";
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        if (K_H > H_in || K_W > W_in) {
            std::cerr << "Filter size (" << K_H << "x" << K_W << ") must not exceed input size ("
                      << H_in << "x" << W_in << ").\n";
            return EXIT_FAILURE;
        }
    }

    const int H_out = H_in - K_H + 1;
    const int W_out = W_in - K_W + 1;

    const size_t input_elems = static_cast<size_t>(N) * C * H_in * W_in;
    const size_t filter_elems = static_cast<size_t>(M) * C * K_H * K_W;
    const size_t output_elems = static_cast<size_t>(N) * M * H_out * W_out;

    double input_mb = input_elems * sizeof(float) / 1024.0 / 1024.0;
    double filter_mb = filter_elems * sizeof(float) / 1024.0 / 1024.0;
    double output_mb = output_elems * sizeof(float) / 1024.0 / 1024.0;

    std::cout << "========================================" << std::endl;
    std::cout << "CNN Convolution Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Batch (N):        " << N << std::endl;
    std::cout << "Out channels (M): " << M << std::endl;
    std::cout << "In channels (C):  " << C << std::endl;
    std::cout << "Input  (H x W):   " << H_in << " x " << W_in << std::endl;
    std::cout << "Filter (H x W):   " << K_H << " x " << K_W << std::endl;
    std::cout << "Output (H x W):   " << H_out << " x " << W_out << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Input  size:  " << input_mb << " MB" << std::endl;
    std::cout << "Filter size:  " << filter_mb << " MB" << std::endl;
    std::cout << "Output size:  " << output_mb << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<float> h_X(input_elems);
    std::vector<float> h_W(filter_elems);
    std::vector<float> h_ref(output_elems, 0.0f);
    std::vector<float> h_out(output_elems, 0.0f);

    init_data(h_X);
    init_data(h_W);

    // CPU baseline
    std::cout << "\n[Running CPU baseline...]" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_cnn(h_X.data(), h_ref.data(), h_W.data(), N, M, C, H_in, W_in, H_out, W_out, K_H, K_W);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    double cpu_gflops = compute_gflops(cpu_ms, N, M, C, H_out, W_out, K_H, K_W);
    std::cout << "[CPU] " << cpu_ms << " ms, " << cpu_gflops << " GFLOPS" << std::endl;
    PerfStats cpu_stats{cpu_ms, cpu_gflops, true, 1.0};

    // Allocate device memory
    float* d_X;
    float* d_W;
    float* d_out;
    CHECK_CUDA(cudaMalloc(&d_X, input_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W, filter_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, output_elems * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), input_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W, h_W.data(), filter_elems * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Native CUDA kernel
    std::cout << "\n[Running Native CUDA kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, output_elems * sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start));
    launch_native(d_X, d_out, d_W, N, M, C, H_in, W_in, H_out, W_out, K_H, K_W);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float native_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&native_ms, start, stop));
    CHECK_CUDA(
        cudaMemcpy(h_out.data(), d_out, output_elems * sizeof(float), cudaMemcpyDeviceToHost));
    bool native_ok = validate_output(h_ref, h_out);
    double native_gflops = compute_gflops(native_ms, N, M, C, H_out, W_out, K_H, K_W);
    std::cout << "[Native CUDA] " << native_ms << " ms, " << native_gflops << " GFLOPS "
              << (native_ok ? "PASS" : "FAIL") << std::endl;
    PerfStats native_stats{static_cast<double>(native_ms), native_gflops, native_ok,
                           compute_speedup(cpu_ms, native_ms)};

    // unroll CUDA kernel
    std::cout << "\n[Running Unroll CUDA kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, output_elems * sizeof(float)));
    CHECK_CUDA(cudaEventRecord(start));
    launch_unroll(d_X, d_out, d_W, N, M, C, H_in, W_in, H_out, W_out, K_H, K_W);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float unroll_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&unroll_ms, start, stop));
    CHECK_CUDA(
        cudaMemcpy(h_out.data(), d_out, output_elems * sizeof(float), cudaMemcpyDeviceToHost));
    bool unroll_ok = validate_output(h_ref, h_out);
    double unroll_gflops = compute_gflops(unroll_ms, N, M, C, H_out, W_out, K_H, K_W);
    std::cout << "[Unroll CUDA] " << unroll_ms << " ms, " << unroll_gflops << " GFLOPS "
              << (unroll_ok ? "PASS" : "FAIL") << std::endl;
    PerfStats unroll_stats{static_cast<double>(unroll_ms), unroll_gflops, unroll_ok,
                           compute_speedup(cpu_ms, unroll_ms)};

    // cuDNN Implicit GEMM
    std::cout << "\n[Running cuDNN Implicit GEMM...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, output_elems * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start));
    launch_cudnn_conv(d_X, d_out, d_W, N, M, C, H_in, W_in, H_out, W_out, K_H, K_W);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float cudnn_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&cudnn_ms, start, stop));
    CHECK_CUDA(
        cudaMemcpy(h_out.data(), d_out, output_elems * sizeof(float), cudaMemcpyDeviceToHost));

    bool cudnn_ok = validate_output(h_ref, h_out);
    double cudnn_gflops = compute_gflops(cudnn_ms, N, M, C, H_out, W_out, K_H, K_W);

    std::cout << "[cuDNN] " << cudnn_ms << " ms, " << cudnn_gflops << " GFLOPS "
              << (cudnn_ok ? "PASS" : "FAIL") << std::endl;

    PerfStats cudnn_stats{static_cast<double>(cudnn_ms), cudnn_gflops, cudnn_ok,
                          compute_speedup(cpu_ms, cudnn_ms)};

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_W));
    CHECK_CUDA(cudaFree(d_out));

    print_performance_table(cpu_stats, native_stats, unroll_stats, cudnn_stats);

    std::cout << "\nBenchmark completed successfully!" << std::endl;
    return 0;
}
