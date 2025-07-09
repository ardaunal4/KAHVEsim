// main.cu

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gpu_update.cuh"  // include the kernel declaration only
#include <iomanip>
#include <random>
#include <algorithm>

int main() {
    constexpr int N = 1 << 11;  // 2048 particles

    // 1. Host bellekte parçacıkları oluştur (normal dağılım)
    std::vector<Particle> h_particles(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> pos_dist_xy(0.0f, 0.0001f);
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
    }

    // 2. Sabit elektrik alan (her parçacık için aynı Ex, Ey, Ez)
    std::vector<float> h_fields(N * 3);
    for (int i = 0; i < N; ++i) {
        h_fields[3 * i + 0] = 1e5f;  // Ex
        h_fields[3 * i + 1] = 0.0f;  // Ey
        h_fields[3 * i + 2] = 0.0f;  // Ez
    }

    // 3. GPU belleği ayır
    Particle* d_particles;
    float* d_fields;
    cudaMalloc(&d_particles, N * sizeof(Particle));
    cudaMalloc(&d_fields, N * 3 * sizeof(float));

    // 4. Verileri GPU'ya kopyala
    cudaMemcpy(d_particles, h_particles.data(), N * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fields, h_fields.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);

    // 5. Zaman adımı ayarları
    float dt = 1e-11f;
    float halfDt = 0.5f * dt;

    // 6. Kernel konfigürasyonu
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 7. Simülasyon döngüsü: 1000 zaman adımı çalıştır
    for (int step = 0; step < 1000; ++step) {
        updatePositionsKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_particles, d_fields, halfDt, dt, N);
        cudaDeviceSynchronize();
    }

    // 8. GPU'dan sonucu CPU'ya kopyala
    cudaMemcpy(h_particles.data(), d_particles, N * sizeof(Particle), cudaMemcpyDeviceToHost);

    // 9. Sonuçları yazdır (sadece ilk 10 için)
    std::cout << std::fixed << std::setprecision(10);
    for (int i = 0; i < std::min(N, 10); ++i) {
        std::cout << "Particle " << i << ": x=" << h_particles[i].x
                  << ", vx=" << h_particles[i].vx
                  << ", ax=" << h_particles[i].ax << "\n";
    }

    // 10. Bellek temizliği
    cudaFree(d_particles);
    cudaFree(d_fields);

    return 0;
}
