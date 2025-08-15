// gpu_update.cuh
#pragma once

struct Particle {
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
    int active;  // 1 = active, 0 = lost/eliminated
};

__global__
void updatePositionsKernel(Particle* p, const float* fields, float halfDt, float dt, int N);

__global__ 
void checkBoundariesKernel(Particle* p, int N);