// gpu_update.cu
#include "gpu_update.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__
void updatePositionsKernel(Particle* p, const float* fields, float halfDt, float dt, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || p[idx].active == 0) return;  // Skip inactive particles

    const float qmRatio = -1.75882e11f;  // C * kg^(-1)
    const float c = 299792458.0f;        // m/s

    // Leapfrog Step 1: v(n+1/2) = v(n) + a(n) * dt/2
    p[idx].vx += p[idx].ax * halfDt;
    p[idx].vy += p[idx].ay * halfDt;
    p[idx].vz += p[idx].az * halfDt;

    // Position update: x(n+1) = x(n) + v(n+1/2) * dt
    p[idx].x += p[idx].vx * dt;
    p[idx].y += p[idx].vy * dt;
    p[idx].z += p[idx].vz * dt;

    // Current velocity components
    float vx = p[idx].vx;
    float vy = p[idx].vy;
    float vz = p[idx].vz;

    // Relativistic calculations
    float beta_x = vx / c;
    float beta_y = vy / c;
    float beta_z = vz / c;
    float beta2 = beta_x * beta_x + beta_y * beta_y + beta_z * beta_z;
    float gamma = rsqrtf(1.0f - beta2);  // rsqrtf = 1/sqrt(), so this gives 1/sqrt(1-β²) = γ

    // Electric field components
    float Ex = fields[3 * idx + 0];
    float Ey = fields[3 * idx + 1];
    float Ez = fields[3 * idx + 2];

    // Relativistic momentum equation implementation
    // Sequential calculation matching Python behavior exactly
    
    float gamma2 = gamma * gamma;
    
    // X-component acceleration (uses OLD ay, az values)
    float cross_term_x = gamma2 * (beta_y * p[idx].ay + beta_z * p[idx].az) * beta_x;
    p[idx].ax = (qmRatio * Ex / gamma - cross_term_x) / (1.0f + gamma2 * beta_x * beta_x);
    
    // Y-component acceleration (uses NEW ax, OLD az)  
    float cross_term_y = gamma2 * (beta_x * p[idx].ax + beta_z * p[idx].az) * beta_y;
    p[idx].ay = (qmRatio * Ey / gamma - cross_term_y) / (1.0f + gamma2 * beta_y * beta_y);
    
    // Z-component acceleration (uses NEW ax, NEW ay)
    float cross_term_z = gamma2 * (beta_x * p[idx].ax + beta_y * p[idx].ay) * beta_z;
    p[idx].az = (qmRatio * Ez / gamma - cross_term_z) / (1.0f + gamma2 * beta_z * beta_z);

    // Leapfrog Step 2: v(n+1) = v(n+1/2) + a(n+1) * dt/2
    p[idx].vx += p[idx].ax * halfDt;
    p[idx].vy += p[idx].ay * halfDt;
    p[idx].vz += p[idx].az * halfDt;
}

__global__ 
void checkBoundariesKernel(Particle* p, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N || p[idx].active == 0) return;

    float x = p[idx].x;
    float y = p[idx].y;
    float z = p[idx].z;
    
    // Boundary checking matching Python implementation
    bool isValid = true;
    
    // Region 1: z < -0.168 m
    if (z < -0.168f) {
        if (fabsf(x) > 0.04f || fabsf(y) > 0.04f) {
            isValid = false;
        }
    }
    // Region 2: -0.168 <= z <= -0.05 m
    else if (z >= -0.168f && z <= -0.05f) {
        if (fabsf(x) > 0.04f || fabsf(y) > 0.02f) {
            isValid = false;
        }
    }
    // Region 3: -0.05 < z < 0.324 m
    else if (z > -0.05f && z < 0.324f) {
        float y_upper = 0.04f + (z + 0.049f) * (-0.2672f);
        float y_lower = -0.04f + (z + 0.049f) * (-0.2672f);
        if (fabsf(x) > 0.04f || y > y_upper || y < y_lower) {
            isValid = false;
        }
    }
    // Parallel plate boundary check (0.324 <= z <= 0.438)
    else if (z >= 0.324f && z <= 0.438f) {
        float y_upper = -0.075f + (z - 0.313f) * (-0.2672f);
        float y_lower = -0.118f + (z - 0.313f) * (-0.2672f);
        if (fabsf(x) > 0.04f || y > y_upper || y < y_lower) {
            isValid = false;
        }
    }
    // Region 4: 0.438 < z <= 1.5 m
    else if (z > 0.438f && z <= 1.5f) {
        if (fabsf(x) > 0.04f || y > (-0.134f + 0.039f) || y < (-0.134f - 0.040f)) {
            isValid = false;
        }
    }
    
    // Set particle as inactive if it hits boundaries
    if (!isValid) {
        p[idx].active = 0;
    }
}