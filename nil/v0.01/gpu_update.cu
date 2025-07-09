// gpu_update.cu
#include "gpu_update.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__
void updatePositionsKernel(Particle* p, const float* fields, float halfDt, float dt, int N)
 {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float qmRatio = -1.75882e11f;
    const float c = 299792458.0f;

    p[idx].vx += p[idx].ax * halfDt;
    p[idx].vy += p[idx].ay * halfDt;
    p[idx].vz += p[idx].az * halfDt;

    p[idx].x += p[idx].vx * dt;
    p[idx].y += p[idx].vy * dt;
    p[idx].z += p[idx].vz * dt;

    float vx = p[idx].vx, vy = p[idx].vy, vz = p[idx].vz;
    float beta2 = (vx*vx + vy*vy + vz*vz) / (c*c);
    float gamma = rsqrtf(1.0f - beta2);

    float Ex = fields[3 * idx + 0];
    float Ey = fields[3 * idx + 1];
    float Ez = fields[3 * idx + 2];

    p[idx].ax = (qmRatio * Ex) / gamma;
    p[idx].ay = (qmRatio * Ey) / gamma;
    p[idx].az = (qmRatio * Ez) / gamma;

    p[idx].vx += p[idx].ax * halfDt;
    p[idx].vy += p[idx].ay * halfDt;
    p[idx].vz += p[idx].az * halfDt;
}
