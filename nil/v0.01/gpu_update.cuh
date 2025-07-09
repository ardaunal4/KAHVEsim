// gpu_update.cuh
#pragma once

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
};

__global__
void updatePositionsKernel(Particle* p, const float* fields, float halfDt, float dt, int N);

