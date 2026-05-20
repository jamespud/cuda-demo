#include <cstdlib>
#include <exception>
#include <string>

#include "../include/common.cuh"
#include "../include/reduction.cuh"

constexpr size_t kDefaultCount = 4u * 1024u * 1024u;

struct PerfStats {
    double time_ms;
    bool passed;
    double speedup;
};

void print_performance_table(const PerfStats& cpu, const PerfStats& native, const PerfStats& shared,
                             const PerfStats& coarsened) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Version" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup" << "Status" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "CPU" << std::setw(15) << std::fixed
              << std::setprecision(3) << cpu.time_ms << std::setw(15) << "1.00x" << "N/A"
              << std::endl;
    std::cout << std::left << std::setw(15) << "Native CUDA" << std::setw(15) << native.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << native.speedup << "x"
              << (native.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << std::left << std::setw(15) << "Shared CUDA" << std::setw(15) << shared.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << shared.speedup << "x"
              << (shared.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << std::left << std::setw(15) << "Coarsened CUDA" << std::setw(15)
              << coarsened.time_ms << std::setw(15) << std::fixed << std::setprecision(2)
              << coarsened.speedup << "x" << (coarsened.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "========================================" << std::endl;
}

double compute_speedup(double baseline_ms, double measured_ms) {
    return measured_ms > 0.0 ? baseline_ms / measured_ms : 0.0;
}

bool parse_positive_arg(const char* text, size_t& value) {
    try {
        std::string token(text);
        size_t consumed = 0;
        long long parsed = std::stoll(token, &consumed);
        if (consumed != token.size() || parsed <= 0) return false;
        value = static_cast<size_t>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [count]" << std::endl;
    std::cout << "If no arguments are provided, the default size " << kDefaultCount << " is used."
              << std::endl;
}

void init_data(std::vector<unsigned int>& data) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<unsigned int> dist(0, 255);
    for (auto& value : data) {
        value = dist(rng);
    }
}

bool validate_reduction(const std::vector<unsigned long long>& ref,
                        const std::vector<unsigned long long>& out) {
    if (ref[0] != out[0]) {
        std::cerr << "Mismatch: ref=" << ref[0] << " got=" << out[0] << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    size_t count = kDefaultCount;

    if (argc != 1 && argc != 2) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 2) {
        if (!parse_positive_arg(argv[1], count)) {
            std::cerr << "Invalid argument. count must be a positive integer." << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    size_t input_size = count * sizeof(unsigned int);

    std::cout << "========================================" << std::endl;
    std::cout << "Reduction Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Count:    " << count << std::endl;
    std::cout << "Input Size:  " << input_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Output Size: " << sizeof(unsigned long long) << " bytes" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<unsigned int> h_in(count);
    std::vector<unsigned long long> h_ref(1, 0);
    std::vector<unsigned long long> h_out(1, 0);

    init_data(h_in);

    std::cout << "\n[Running CPU baseline...]" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_reduction(h_in.data(), h_ref.data(), count);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "[CPU] " << cpu_ms << " ms, result = " << h_ref[0] << std::endl;
    PerfStats cpu_stats{cpu_ms, true, 1.0};

    unsigned int* d_in;
    unsigned long long* d_out;
    CHECK_CUDA(cudaMalloc(&d_in, input_size));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(unsigned long long)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), input_size, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // native cuda
    std::cout << "\n[Running Native CUDA kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    CHECK_CUDA(cudaEventRecord(start));
    launch_native(d_in, d_out, count);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float native_ms;
    CHECK_CUDA(cudaEventElapsedTime(&native_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    bool native_ok = validate_reduction(h_ref, h_out);
    std::cout << "[Native CUDA] " << native_ms << " ms " << (native_ok ? "PASS" : "FAIL")
              << ", result = " << h_out[0] << std::endl;
    PerfStats native_stats{static_cast<double>(native_ms), native_ok,
                           compute_speedup(cpu_ms, native_ms)};

    // shared cuda
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), input_size, cudaMemcpyHostToDevice));
    std::cout << "\n[Running Shared CUDA kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    CHECK_CUDA(cudaEventRecord(start));
    launch_shared(d_in, d_out, count);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float shared_ms;
    CHECK_CUDA(cudaEventElapsedTime(&shared_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    bool shared_ok = validate_reduction(h_ref, h_out);
    std::cout << "[Shared CUDA] " << shared_ms << " ms " << (shared_ok ? "PASS" : "FAIL")
              << ", result = " << h_out[0] << std::endl;
    PerfStats shared_stats{static_cast<double>(shared_ms), shared_ok,
                           compute_speedup(cpu_ms, shared_ms)};

    // coarsened cuda
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), input_size, cudaMemcpyHostToDevice));
    std::cout << "\n[Running Coarsened CUDA kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(unsigned long long)));
    CHECK_CUDA(cudaEventRecord(start));
    launch_coarsened(d_in, d_out, count);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float coarsened_ms;
    CHECK_CUDA(cudaEventElapsedTime(&coarsened_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    bool coarsened_ok = validate_reduction(h_ref, h_out);
    std::cout << "[Coarsened CUDA] " << coarsened_ms << " ms " << (coarsened_ok ? "PASS" : "FAIL")
              << ", result = " << h_out[0] << std::endl;
    PerfStats coarsened_stats{static_cast<double>(coarsened_ms), coarsened_ok,
                              compute_speedup(cpu_ms, coarsened_ms)};

    print_performance_table(cpu_stats, native_stats, shared_stats, coarsened_stats);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    std::cout << "\nBenchmark completed successfully!" << std::endl;
    return 0;
}
