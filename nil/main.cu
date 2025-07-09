// Rhodotron Beam Chopper Simulation - Main Program with CSV Output & Time-Varying Fields

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gpu_update.cuh"  
#include "csv_reader.h"    
#include "csv_writer.h"    
#include "pulser.h"        
#include <iomanip>
#include <random>
#include <algorithm>
#include <chrono>
#include <cfloat>          
#include <cmath>           

__host__ __device__ 
int findNearestFieldIndex(const EFieldPoint* fieldPoints, int numPoints, 
                         float px, float py, float pz) {
    float minDist = FLT_MAX;
    int bestIndex = 0;
    
    for (int i = 0; i < numPoints; i++) {
        float dx = fieldPoints[i].x - px;
        float dy = fieldPoints[i].y - py;
        float dz = fieldPoints[i].z - pz;
        float dist = dx*dx + dy*dy + dz*dz;
        
        if (dist < minDist) {
            minDist = dist;
            bestIndex = i;
        }
    }
    
    return bestIndex;
}

int main() {
    constexpr int N = 1 << 11;  // 2048 particles
    
    std::cout << "=== Rhodotron Beam Chopper Simulation ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Load electric field data
    std::cout << "\n1. Loading electric field data..." << std::endl;
    std::vector<EFieldPoint> h_fieldPoints = CSVReader::loadElectricField("efield3.csv", 100000);
    
    if (h_fieldPoints.empty()) {
        std::cerr << "Warning: No field data loaded." << std::endl;
        return -1;
    } else {
        std::cout << "Successfully loaded " << h_fieldPoints.size() << " field points!" << std::endl;
    }

    // Initialize output files
    std::cout << "\n2. Initializing output systems..." << std::endl;
    std::string outputFile = "simulation_results.csv";
    std::string statsFile = "simulation_statistics.csv";
    
    if (!CSVWriter::initializeOutputFile(outputFile)) {
        std::cerr << "Failed to initialize output file. Exiting." << std::endl;
        return -1;
    }
    
    // Initialize pulser
    Pulser pulser(2.5f, 10.0f, 0.131184f);

    // Initialize particles
    std::cout << "\n3. Initializing particles..." << std::endl;
    std::vector<Particle> h_particles(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Same distributions as Python
    std::normal_distribution<float> pos_dist_xy(0.0f, 0.0001f);     // ±0.1mm 
    std::normal_distribution<float> pos_dist_z(-1.5f, 1.0f);       
    
    float c = 299792458.0f;           
    float vz_mean = 0.3941f * c;      
    std::normal_distribution<float> vel_dist_z(vz_mean, vz_mean * 1e-6f);

    for (int i = 0; i < N; ++i) {
        h_particles[i].x = pos_dist_xy(gen);    
        h_particles[i].y = pos_dist_xy(gen);    
        h_particles[i].z = pos_dist_z(gen);     
        h_particles[i].vx = 0.0f;               
        h_particles[i].vy = 0.0f;               
        h_particles[i].vz = vel_dist_z(gen);    
        h_particles[i].ax = 0.0f;               
        h_particles[i].ay = 0.0f;               
        h_particles[i].az = 0.0f;               
        h_particles[i].active = 1;              
    }

    std::cout << "Initial beam parameters:" << std::endl;
    std::cout << "  Mean Z velocity: " << vz_mean << " m/s (β = " << vz_mean/c << ")" << std::endl;

    // GPU memory allocation
    std::cout << "\n4. Allocating GPU memory..." << std::endl;
    Particle* d_particles;
    float* d_fields;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMalloc(&d_fields, N * 3 * sizeof(float));

    cudaMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);

    // Simulation parameters
    float dt = 1e-11f;          // 0.01 nanosecond
    float halfDt = 0.5f * dt;   
    float time = 0.0f;          
    float time_ns = 0.0f;       

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    std::vector<float> h_fields(N * 3);

    // Main simulation loop
    std::cout << "\n5. Starting simulation loop..." << std::endl;
    
    const int totalSteps = 1600;  // Same as Python iteration count
    const int saveInterval = 10;  // Save every 10th timestep
    
    for (int step = 0; step < totalSteps; ++step) {
        
        time = step * dt;           
        time_ns = time * 1e9f;      
        
        // Field interpolation - NO AMPLIFICATION like Python
        for (int i = 0; i < N; ++i) {
            int fieldIndex = findNearestFieldIndex(h_fieldPoints.data(), h_fieldPoints.size(),
                                                  h_particles[i].x, h_particles[i].y, h_particles[i].z);
            
            // NO AMPLIFICATION - use fields directly like Python
            h_fields[3 * i + 0] = h_fieldPoints[fieldIndex].Ex;  // Ex
            h_fields[3 * i + 1] = h_fieldPoints[fieldIndex].Ey;  // Ey
            h_fields[3 * i + 2] = h_fieldPoints[fieldIndex].Ez;  // Ez
        }
        
        // Apply time-varying pulse
        pulser.applyPulse(h_fieldPoints, time_ns, h_fields.data(), N, h_particles.data());
        
        // DEBUG: Print field values ​​of first 3 particles
        if (step % 100 == 0 && step < 500) {
            std::cout << "  DEBUG Fields at step " << step << ":" << std::endl;
            for (int debug_i = 0; debug_i < 3; debug_i++) {
                std::cout << "    Particle " << debug_i 
                          << ": Ex=" << h_fields[3 * debug_i + 0] 
                          << ", Ey=" << h_fields[3 * debug_i + 1] 
                          << ", Ez=" << h_fields[3 * debug_i + 2] << std::endl;
            }
            std::cout << "    Pulse amplitude: " << pulser.calculatePulseAmplitude(time_ns, 0.0f) << std::endl;
        }
        
        cudaMemcpy(d_fields, h_fields.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
        
        // GPU calculation
        updatePositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_particles, d_fields, halfDt, dt, N);
        
        checkBoundariesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, N);
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error at step " << step << ": " << cudaGetErrorString(err) << std::endl;
            break;
        }
        
        cudaMemcpy(h_particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);
        
        // Count active particles
        int activeParticles = 0;
        for (int i = 0; i < N; ++i) {
            if (h_particles[i].active) activeParticles++;
        }
        
        // Save data - every 10th timestep, all particles
        if (step % saveInterval == 0 || step == 0) {
            // Save all particles
            CSVWriter::saveTimestepSubset(outputFile, h_particles.data(), N, time, step, 1);  // saveInterval=1 means all particles
        }
        
        // Progress monitoring
        if (step % 100 == 0) {
            std::cout << "Step " << step << ", Time: " << time_ns << " ns" << std::endl;
            std::cout << "  Active particles: " << activeParticles << "/" << N 
                      << " (" << (100.0f * activeParticles / N) << "%)" << std::endl;
            std::cout << "  Particle 0: x=" << h_particles[0].x 
                      << ", vx=" << h_particles[0].vx 
                      << ", z=" << h_particles[0].z << std::endl;
        }
        
        cudaMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);
    }

    // Results
    std::cout << "\n=== Final Results ===" << std::endl;
    for (int i = 0; i < std::min(N, 10); ++i) {
        std::cout << "Particle " << i << ": "
                  << "pos=(" << h_particles[i].x << ", " << h_particles[i].y << ", " << h_particles[i].z << ") "
                  << "vel=(" << h_particles[i].vx << ", " << h_particles[i].vy << ", " << h_particles[i].vz << ")" 
                  << std::endl;
    }

    // Performance statistics
    auto endTime = std::chrono::high_resolution_clock::now();
    auto executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "\n=== Performance Statistics ===" << std::endl;
    std::cout << "Total execution time: " << executionDuration.count() << " ms" << std::endl;
    std::cout << "Particles simulated: " << N << std::endl;
    std::cout << "Time steps: " << totalSteps << std::endl;

    float totalSimTime = totalSteps * dt;
    CSVWriter::saveStatistics(statsFile, h_particles.data(), N, totalSimTime, totalSteps, executionDuration.count());

    // Cleanup
    cudaFree(d_particles);
    cudaFree(d_fields);

    std::cout << "\nSimulation completed successfully!" << std::endl;
    std::cout << "Results saved to: " << outputFile << std::endl;
    
    return 0;
}