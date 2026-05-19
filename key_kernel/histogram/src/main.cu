#include <exception>
#include <string>

#include "../include/common.cuh"
#include "../include/histogram.cuh"

struct PerfStats {
    double time_ms;
    bool passed;
    double speedup;
};

void print_performance_table(const PerfStats& cpu, const PerfStats& naive, const PerfStats& shared,
                             const PerfStats& coarsened, const PerfStats& registered) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Performance Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::left << std::setw(15) << "Version" << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Speedup" << "Status" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(15) << "CPU" << std::setw(15) << std::fixed
              << std::setprecision(3) << cpu.time_ms << std::setw(15) << "1.00x" << "N/A" << std::endl;
    std::cout << std::left << std::setw(15) << "Naive CUDA" << std::setw(15) << naive.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << naive.speedup << "x"
              << (naive.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << std::left << std::setw(15) << "Shared Mem" << std::setw(15) << shared.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << shared.speedup << "x"
              << (shared.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << std::left << std::setw(15) << "Coarsened" << std::setw(15) << coarsened.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << coarsened.speedup << "x"
              << (coarsened.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << std::left << std::setw(15) << "Registered" << std::setw(15) << registered.time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << registered.speedup << "x"
              << (registered.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "========================================" << std::endl;
}

bool parse_positive_arg(const char* text, int& value) {
    try {
        std::string token(text);
        size_t consumed = 0;
        int parsed = std::stoi(token, &consumed);
        if (consumed != token.size() || parsed <= 0) return false;
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [channels width bins]" << std::endl;
    std::cout << "If no arguments are provided, the default size 1024 4096 256 is used." << std::endl;
}

void init_data(std::vector<unsigned char>& data, int bins) {
    std::mt19937 rng(42);
    int upper = bins < 256 ? bins : 256;
    std::uniform_int_distribution<int> dist(0, upper - 1);
    for (auto& v : data) {
        v = static_cast<unsigned char>(dist(rng));
    }
}

bool validate_histogram(const std::vector<unsigned int>& ref,
                        const std::vector<unsigned int>& out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (ref[i] != out[i]) {
            std::cerr << "Mismatch at index " << i << ": ref=" << ref[i] << " got=" << out[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    int channels = 1024;
    int width = 4096;
    int bins = 256;

    if (argc != 1 && argc != 4) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 4) {
        if (!parse_positive_arg(argv[1], channels) || !parse_positive_arg(argv[2], width) ||
            !parse_positive_arg(argv[3], bins)) {
            std::cerr << "Invalid arguments. channels, width, and bins must be positive integers." << std::endl;
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    size_t input_count = static_cast<size_t>(channels) * static_cast<size_t>(width);
    size_t hist_count = static_cast<size_t>(channels) * static_cast<size_t>(bins);
    size_t input_size = input_count * sizeof(unsigned char);
    size_t hist_size = hist_count * sizeof(unsigned int);

    std::cout << "========================================" << std::endl;
    std::cout << "Histogram Kernel Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Channels: " << channels << std::endl;
    std::cout << "Width:    " << width << std::endl;
    std::cout << "Bins:     " << bins << std::endl;
    std::cout << "Input Size:  " << input_size / 1024.0 << " KB" << std::endl;
    std::cout << "Hist Size:   " << hist_size / 1024.0 << " KB" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<unsigned char> h_in(input_count);
    std::vector<unsigned int> h_ref(hist_count, 0);
    std::vector<unsigned int> h_out(hist_count);

    init_data(h_in, bins);

    std::cout << "\n[Running CPU baseline...]" << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_histogram(h_in.data(), h_ref.data(), channels, width, bins);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    std::cout << "[CPU] " << cpu_ms << " ms" << std::endl;
    PerfStats cpu_stats{cpu_ms, true, 1.0};

    unsigned char* d_in;
    unsigned int* d_out;
    CHECK_CUDA(cudaMalloc(&d_in, input_size));
    CHECK_CUDA(cudaMalloc(&d_out, hist_size));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), input_size, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // --- Naive ---
    std::cout << "\n[Running Naive CUDA kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, hist_size));
    CHECK_CUDA(cudaEventRecord(start));
    launch_native(d_in, d_out, channels, width, bins);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float naive_ms;
    CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, hist_size, cudaMemcpyDeviceToHost));
    bool naive_ok = validate_histogram(h_ref, h_out, hist_count);
    std::cout << "[Naive CUDA] " << naive_ms << " ms " << (naive_ok ? "PASS" : "FAIL") << std::endl;
    PerfStats naive_stats{static_cast<double>(naive_ms), naive_ok, cpu_ms / naive_ms};

    // --- Shared ---
    std::cout << "\n[Running Shared Memory kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, hist_size));
    CHECK_CUDA(cudaEventRecord(start));
    launch_shared(d_in, d_out, channels, width, bins);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float shared_ms;
    CHECK_CUDA(cudaEventElapsedTime(&shared_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, hist_size, cudaMemcpyDeviceToHost));
    bool shared_ok = validate_histogram(h_ref, h_out, hist_count);
    std::cout << "[Shared Mem] " << shared_ms << " ms " << (shared_ok ? "PASS" : "FAIL") << std::endl;
    PerfStats shared_stats{static_cast<double>(shared_ms), shared_ok, cpu_ms / shared_ms};

    // --- Coarsened ---
    std::cout << "\n[Running Coarsened kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, hist_size));
    CHECK_CUDA(cudaEventRecord(start));
    launch_coarsened(d_in, d_out, channels, width, bins);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float coarsened_ms;
    CHECK_CUDA(cudaEventElapsedTime(&coarsened_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, hist_size, cudaMemcpyDeviceToHost));
    bool coarsened_ok = validate_histogram(h_ref, h_out, hist_count);
    std::cout << "[Coarsened] " << coarsened_ms << " ms " << (coarsened_ok ? "PASS" : "FAIL") << std::endl;
    PerfStats coarsened_stats{static_cast<double>(coarsened_ms), coarsened_ok, cpu_ms / coarsened_ms};

    // --- Registered ---
    std::cout << "\n[Running Registered kernel...]" << std::endl;
    CHECK_CUDA(cudaMemset(d_out, 0, hist_size));
    CHECK_CUDA(cudaEventRecord(start));
    launch_registered(d_in, d_out, channels, width, bins);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float registered_ms;
    CHECK_CUDA(cudaEventElapsedTime(&registered_ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, hist_size, cudaMemcpyDeviceToHost));
    bool registered_ok = validate_histogram(h_ref, h_out, hist_count);
    std::cout << "[Registered] " << registered_ms << " ms " << (registered_ok ? "PASS" : "FAIL") << std::endl;
    PerfStats registered_stats{static_cast<double>(registered_ms), registered_ok, cpu_ms / registered_ms};

    print_performance_table(cpu_stats, naive_stats, shared_stats, coarsened_stats, registered_stats);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    std::cout << "\nBenchmark completed successfully!" << std::endl;
    return 0;
}
