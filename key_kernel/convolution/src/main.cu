#include "conv.cuh"
#include "timer.cuh"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Config {
    int width = 4096;
    int height = 4096;
    int filter_size = 5;
};

Config parse_config(int argc, char* argv[]) {
    Config config;

    if (argc > 4) {
        throw std::runtime_error("Usage: main [width] [height] [filtersize]");
    }

    if (argc > 1) {
        config.width = std::atoi(argv[1]);
    }

    if (argc > 2) {
        config.height = std::atoi(argv[2]);
    }

    if (argc > 3) {
        config.filter_size = std::atoi(argv[3]);
    }

    if (config.width <= 0 || config.height <= 0 || config.filter_size <= 0 ||
        config.filter_size > MAX_FILTER_SIZE) {
        throw std::runtime_error(
            "width and height must be positive, and filtersize must be between 1 and 32");
    }

    return config;
}

std::string format_time_line(const std::string& label, double ms) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << label << ms << " ms";
    return stream.str();
}

template <typename Fn>
double measure_host_time(Fn&& fn) {
    const auto start = std::chrono::high_resolution_clock::now();
    fn();
    const auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

int main(int argc, char* argv[]) {
    Config config;

    try {
        config = parse_config(argc, argv);
    } catch (const std::runtime_error& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    LOG_INFO("Convolution Benchmark Start");

    const int width = config.width;
    const int height = config.height;
    const int filter_size = config.filter_size;

    const size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
    const size_t bytes = size * sizeof(float);

    std::vector<float> h_input(size);
    std::vector<float> h_output_cpu(size);
    std::vector<float> h_output_gpu(size);

    std::vector<float> h_filter(filter_size * filter_size);

    std::mt19937 rng(0);

    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for (auto& x : h_input) {
        x = dist(rng);
    }

    for (auto& x : h_filter) {
        x = dist(rng);
    }

    double cpu_ms = measure_host_time([&] {
        conv_cpu(h_input.data(), h_filter.data(), h_output_cpu.data(), width, height,
                 filter_size);
    });

    LOG_INFO(format_time_line("CPU time: ", cpu_ms));

    float* d_input = nullptr;
    float* d_output = nullptr;
    float* d_filter = nullptr;

    double allocation_ms = measure_host_time([&] {
        CHECK_CUDA(cudaMalloc(&d_input, bytes));
        CHECK_CUDA(cudaMalloc(&d_output, bytes));
        CHECK_CUDA(cudaMalloc(&d_filter, filter_size * filter_size * sizeof(float)));
    });

    LOG_INFO(format_time_line("Allocation time: ", allocation_ms));

    double copy_to_gpu_ms = measure_host_time([&] {
        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(),
                              filter_size * filter_size * sizeof(float),
                              cudaMemcpyHostToDevice));
        copy_filter_to_constant(h_filter.data(), filter_size);
    });

    LOG_INFO(format_time_line("Copy to GPU time: ", copy_to_gpu_ms));

    GpuTimer timer;
    bool success = true;
    double copy_from_gpu_ms = 0.0;

    timer.start();
    launch_conv_native(d_input, d_filter, d_output, width, height, filter_size);
    float naive_kernel_ms = timer.stop();
    LOG_INFO(format_time_line("GPU kernel time (without tiling): ", naive_kernel_ms));

    copy_from_gpu_ms += measure_host_time([&] {
        CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    });

    if (!check_result(h_output_cpu.data(), h_output_gpu.data(), static_cast<int>(size))) {
        success = false;
    }

    timer.start();
    launch_conv_tiled(d_input, d_output, width, height, filter_size);
    float tiled_kernel_ms = timer.stop();
    LOG_INFO(format_time_line("GPU kernel time (with tiling): ", tiled_kernel_ms));

    copy_from_gpu_ms += measure_host_time([&] {
        CHECK_CUDA(cudaMemcpy(h_output_gpu.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    });

    if (!check_result(h_output_cpu.data(), h_output_gpu.data(), static_cast<int>(size))) {
        success = false;
    }

    timer.start();
    copy_filter_to_constant(h_filter.data(), filter_size);
    float constant_kernel_ms = timer.stop();
    LOG_INFO(format_time_line("GPU kernel time (with constant): ", constant_kernel_ms));

    LOG_INFO(format_time_line("Copy from GPU time: ", copy_from_gpu_ms));

    double deallocation_ms = measure_host_time([&] {
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));
        CHECK_CUDA(cudaFree(d_filter));
    });

    LOG_INFO(format_time_line("Deallocation time: ", deallocation_ms));
    LOG_INFO("Benchmark Finished");

    return success ? 0 : 1;
}