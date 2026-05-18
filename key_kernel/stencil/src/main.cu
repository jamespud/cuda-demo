#include <exception>
#include <string>

#include "../include/common.cuh"
#include "../include/stencil.cuh"

// Performance statistics structure
struct PerfStats {
    double time_ms;
    bool passed;
    double speedup;
};

// Print performance comparison table
void print_performance_table(const PerfStats& cpu, const PerfStats& naive, const PerfStats& shared,
                             const PerfStats& coarsened) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Version" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup"
              << "Status" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    std::cout << std::left << std::setw(15) << "CPU" << std::setw(15) << std::fixed
              << std::setprecision(3) << cpu.time_ms << std::setw(15) << "1.00x"
              << "N/A" << std::endl;

    std::cout << std::left << std::setw(15) << "Naive CUDA" << std::setw(15) << naive.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << naive.speedup << "x"
              << (naive.passed ? "PASS" : "FAIL") << std::endl;

    std::cout << std::left << std::setw(15) << "Shared Mem" << std::setw(15) << shared.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << shared.speedup << "x"
              << (shared.passed ? "PASS" : "FAIL") << std::endl;

    std::cout << std::left << std::setw(15) << "Coarsened" << std::setw(15) << coarsened.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << coarsened.speedup << "x"
              << (coarsened.passed ? "PASS" : "FAIL") << std::endl;

    std::cout << "========================================" << std::endl;
}

bool parse_dimension_arg(const char* text, int& value) {
    try {
        std::string token(text);
        size_t consumed = 0;
        int parsed = std::stoi(token, &consumed);
        if (consumed != token.size() || parsed <= 0) {
            return false;
        }

        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [nx ny nz]" << std::endl;
    std::cout << "If no arguments are provided, the default size 256 256 256 is used." << std::endl;
}

int main(int argc, char** argv) {
    int nx = 256;
    int ny = 256;
    int nz = 256;

    if (argc != 1 && argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 4) {
        if (!parse_dimension_arg(argv[1], nx) || !parse_dimension_arg(argv[2], ny) ||
            !parse_dimension_arg(argv[3], nz)) {
            std::cerr << "Invalid dimension arguments. nx, ny, and nz must be positive integers."
                      << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    size_t element_count =
        static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    size_t size = element_count * sizeof(float);

    std::cout << "========================================" << std::endl;
    std::cout << "3D Stencil Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Grid Size: " << nx << " x " << ny << " x " << nz << std::endl;
    std::cout << "Total Elements: " << element_count << std::endl;
    std::cout << "Memory Size: " << size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    //-----------------------------------
    // Host allocation
    //-----------------------------------

    std::vector<float> h_in(element_count);
    std::vector<float> h_ref(element_count);
    std::vector<float> h_out(element_count);

    init_data(h_in);

    //-----------------------------------
    // CPU baseline
    //-----------------------------------

    std::cout << "\n[Running CPU baseline...]" << std::endl;

    auto cpu_start = std::chrono::high_resolution_clock::now();

    cpu_stencil(h_in.data(), h_ref.data(), nx, ny, nz);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::cout << "[CPU] " << cpu_ms << " ms" << std::endl;

    PerfStats cpu_stats{cpu_ms, true, 1.0};

    size_t bytes =
        static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz) * sizeof(float);

    //-----------------------------------
    // Device allocation
    //-----------------------------------

    float* d_in;
    float* d_out;

    CHECK_CUDA(cudaMalloc(&d_in, size));
    CHECK_CUDA(cudaMalloc(&d_out, size));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice));

    //-----------------------------------
    // CUDA Event setup
    //-----------------------------------

    cudaEvent_t start, stop;

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    //-----------------------------------
    // Naive CUDA kernel
    //-----------------------------------

    std::cout << "\n[Running Naive CUDA kernel...]" << std::endl;
    
    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    CHECK_CUDA(cudaEventRecord(start));

    launch_native(d_in, d_out, nx, ny, nz);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float naive_ms;

    CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost));

    bool naive_ok = validate_result(h_ref, h_out);

    std::cout << "[Naive CUDA] " << naive_ms << " ms " << (naive_ok ? "PASS" : "FAIL") << std::endl;

    PerfStats naive_stats{static_cast<double>(naive_ms), naive_ok, cpu_ms / naive_ms};

    //-----------------------------------
    // Shared memory kernel
    //-----------------------------------

    std::cout << "\n[Running Shared Memory kernel...]" << std::endl;

    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    CHECK_CUDA(cudaEventRecord(start));

    launch_shared(d_in, d_out, nx, ny, nz);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float shared_ms;

    CHECK_CUDA(cudaEventElapsedTime(&shared_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost));

    bool shared_ok = validate_result(h_ref, h_out);

    std::cout << "[Shared Mem] " << shared_ms << " ms " << (shared_ok ? "PASS" : "FAIL")
              << std::endl;

    PerfStats shared_stats{static_cast<double>(shared_ms), shared_ok, cpu_ms / shared_ms};

    //-----------------------------------
    // Coarsened kernel
    //-----------------------------------

    std::cout << "\n[Running Coarsened kernel...]" << std::endl;

    CHECK_CUDA(cudaMemset(d_out, 0, bytes));

    CHECK_CUDA(cudaEventRecord(start));

    launch_coarsened(d_in, d_out, nx, ny, nz);

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coarsened_ms;

    CHECK_CUDA(cudaEventElapsedTime(&coarsened_ms, start, stop));

    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost));

    bool coarsened_ok = validate_result(h_ref, h_out);

    std::cout << "[Coarsened] " << coarsened_ms << " ms " << (coarsened_ok ? "PASS" : "FAIL")
              << std::endl;

    PerfStats coarsened_stats{static_cast<double>(coarsened_ms), coarsened_ok,
                              cpu_ms / coarsened_ms};

    //-----------------------------------
    // Print performance summary
    //-----------------------------------

    print_performance_table(cpu_stats, naive_stats, shared_stats, coarsened_stats);

    //-----------------------------------
    // Cleanup
    //-----------------------------------

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    std::cout << "\nBenchmark completed successfully!" << std::endl;

    return 0;
}
