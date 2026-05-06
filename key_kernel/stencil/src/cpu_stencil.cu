#include "common.cuh"
#include "stencil.cuh"

// CPU baseline implementation
void cpu_stencil(const float* in, float* out, int nx, int ny, int nz) {
    // For each interior point (i, j, k), compute:
    // out[idx] = kCenter * in[idx] +
    //            kNeighbor * (in[idx-1] + in[idx+1] +
    //                        in[idx-nx] + in[idx+nx] +
    //                        in[idx-nx*ny] + in[idx+nx*ny])
    //
    // Note: Skip boundary points (i=0, i=nx-1, j=0, j=ny-1, k=0, k=nz-1)

    // Placeholder - replace with actual implementation
    for (int k = 1; k < nz - 1; k++) {
        for (int j = 1; j < ny - 1; j++) {
            for (int i = 1; i < nx - 1; i++) {
                int idx = i + j * nx + k * nx * ny;
                out[idx] = kCenter * in[idx] +
                           kNeighbor * (in[idx - 1] + in[idx + 1] + in[idx - nx] + in[idx + nx] +
                                        in[idx - nx * ny] + in[idx + nx * ny]);
            }
        }
    }
}
