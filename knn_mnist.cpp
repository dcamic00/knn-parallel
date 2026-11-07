#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <iomanip>

uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

struct DatasetSoA {
    std::vector<float> all_features; 
    std::vector<uint8_t> labels;     
    size_t num_samples;
    size_t num_features;
};

DatasetSoA read_mnist_images(const std::string& full_path) {
    DatasetSoA dataset;
    std::ifstream file(full_path, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0, num_images = 0, n_rows = 0, n_cols = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_endian(magic_number);
        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        num_images = swap_endian(num_images);
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
        n_rows = swap_endian(n_rows);
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
        n_cols = swap_endian(n_cols);

        dataset.num_samples = num_images;
        dataset.num_features = n_rows * n_cols;
        dataset.all_features.resize(dataset.num_samples * dataset.num_features);

        std::vector<uint8_t> temp_pixels(dataset.num_samples * dataset.num_features);
        file.read(reinterpret_cast<char*>(temp_pixels.data()), temp_pixels.size());

        for (size_t i = 0; i < temp_pixels.size(); ++i) {
            dataset.all_features[i] = static_cast<float>(temp_pixels[i]) / 255.0f;
        }
    } else {
        throw std::runtime_error("Error al abrir: " + full_path);
    }
    return dataset;
}

void read_mnist_labels(const std::string& full_path, DatasetSoA& dataset) {
    std::ifstream file(full_path, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0, num_items = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_endian(magic_number);
        file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
        num_items = swap_endian(num_items);

        dataset.labels.resize(num_items);
        file.read(reinterpret_cast<char*>(dataset.labels.data()), num_items);
    } else {
        throw std::runtime_error("Error al abrir: " + full_path);
    }
}

// ---------------- DISTANCIAS Y SIMD ----------------

float euclidean_distance_sq_simd(const float* img1, const float* img2, size_t num_features) {
    float dist = 0.0f;
    #pragma omp simd reduction(+:dist)
    for (size_t i = 0; i < num_features; ++i) {
        float diff = img1[i] - img2[i];
        dist += diff * diff;
    }
    return dist;
}

float cosine_distance_simd(const float* img1, const float* img2, size_t num_features) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    #pragma omp simd reduction(+:dot, norm1, norm2)
    for (size_t i = 0; i < num_features; ++i) {
        dot += img1[i] * img2[i];
        norm1 += img1[i] * img1[i];
        norm2 += img2[i] * img2[i];
    }
    if (norm1 == 0.0f || norm2 == 0.0f) return 1.0f; // Prevenir división por 0
    return 1.0f - (dot / (std::sqrt(norm1) * std::sqrt(norm2)));
}

// ---------------- CLASIFICACIÓN (MAYORÍA) ----------------

uint8_t get_majority_vote(const std::vector<std::pair<float, uint8_t>>& k_nearest, int k) {
    int counts[10] = {0};
    for (int i = 0; i < k; ++i) {
        counts[k_nearest[i].second]++;
    }
    uint8_t best_label = 0;
    int max_count = -1;
    for (int i = 0; i < 10; ++i) {
        if (counts[i] > max_count) {
            max_count = counts[i];
            best_label = i;
        }
    }
    return best_label;
}

enum DistanceMetric { EUCLIDEAN, COSINE };

// ---------------- RUTINA PRINCIPAL EVALUACIÓN ----------------

double run_knn_evaluation(const DatasetSoA& train, const DatasetSoA& test, int k, int samples, 
                          int threads, omp_sched_t sched, DistanceMetric metric, double& out_accuracy) {
    
    omp_set_num_threads(threads);
    omp_set_schedule(sched, 0); // 0 deja que OpenMP decida el chunk_size óptimo por defecto
    
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // Declaramos el entorno paralelo para todo el bucle
    #pragma omp parallel reduction(+:correct)
    {
        // IMPORTANTE: Instanciar la memoria de distancias UNA SOLA VEZ por hilo.
        // Hacerlo dentro del for mata el rendimiento por culpa de re-allocations masivas.
        std::vector<std::pair<float, uint8_t>> distances(train.num_samples);

        #pragma omp for schedule(runtime)
        for (int i = 0; i < samples; ++i) {
            const float* current_test_img = &test.all_features[i * test.num_features];

            for (size_t j = 0; j < train.num_samples; ++j) {
                float dist;
                if (metric == EUCLIDEAN) {
                    dist = euclidean_distance_sq_simd(current_test_img, &train.all_features[j * train.num_features], train.num_features);
                } else {
                    dist = cosine_distance_simd(current_test_img, &train.all_features[j * train.num_features], train.num_features);
                }
                distances[j].first = dist;
                distances[j].second = train.labels[j];
            }
            
            // Ordenación parcial para obtener los K mejores
            std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                              [](const std::pair<float, uint8_t>& a, const std::pair<float, uint8_t>& b) { return a.first < b.first; });
            
            uint8_t predicted_label = get_majority_vote(distances, k);
            if (predicted_label == test.labels[i]) {
                correct++;
            }
        }
    }
    
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    out_accuracy = (static_cast<double>(correct) / samples) * 100.0;
    return elapsed.count();
}

int main() {
    std::string train_images_path = "train-images.idx3-ubyte";
    std::string train_labels_path = "train-labels.idx1-ubyte";
    std::string test_images_path = "t10k-images.idx3-ubyte";
    std::string test_labels_path = "t10k-labels.idx1-ubyte";
    
    try {
        std::cout << "--- Cargando Dataset MNIST ---" << std::endl;
        DatasetSoA train_data = read_mnist_images(train_images_path);
        read_mnist_labels(train_labels_path, train_data);
        DatasetSoA test_data = read_mnist_images(test_images_path);
        read_mnist_labels(test_labels_path, test_data);
        
        int samples = 10000;
        int optimal_threads = 8;
        int num_runs = 5; // Requisito: Mínimo 5 ejecuciones
        double accuracy = 0.0;
        
        std::cout << "\n1. Evaluacion de Precision (Accuracy) con K=1, 5, 10" << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl;
        int ks[] = {1, 5, 10};
        
        for (int k : ks) {
            run_knn_evaluation(train_data, test_data, k, samples, optimal_threads, omp_sched_static, EUCLIDEAN, accuracy);
            std::cout << "K=" << std::left << std::setw(2) << k << " | Euclidiana | Accuracy: " << accuracy << "%" << std::endl;
            
            run_knn_evaluation(train_data, test_data, k, samples, optimal_threads, omp_sched_static, COSINE, accuracy);
            std::cout << "K=" << std::left << std::setw(2) << k << " | Coseno     | Accuracy: " << accuracy << "%" << std::endl;
        }

        std::cout << "\n2. Iniciando pruebas de Scheduling OpenMP (k=5, 8 Hilos)..." << std::endl;
        std::ofstream sched_file("scheduling_results.csv");
        sched_file << "Schedule_Type,Time_Seconds\n";
        
        omp_sched_t schedules[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
        std::string sched_names[] = {"static", "dynamic", "guided"};
        
        for (int i = 0; i < 3; i++) {
            std::cout << "-> Probando schedule(" << sched_names[i] << ")... ";
            double t_total = 0;
            
            // Warm-up run (descartamos primera ejecucion)
            run_knn_evaluation(train_data, test_data, 5, samples, optimal_threads, schedules[i], EUCLIDEAN, accuracy);
            
            for(int r = 0; r < num_runs; r++) {
                t_total += run_knn_evaluation(train_data, test_data, 5, samples, optimal_threads, schedules[i], EUCLIDEAN, accuracy);
            }
            double t_avg = t_total / num_runs;
            std::cout << t_avg << " seg" << std::endl;
            sched_file << sched_names[i] << "," << t_avg << "\n";
        }
        sched_file.close();
        std::cout << "Pruebas de Scheduling terminadas. ('scheduling_results.csv')" << std::endl;


        std::cout << "\n3. Iniciando pruebas de Escalabilidad Fuerte (Strong Scaling)..." << std::endl;
        std::ofstream scale_file("scaling_results.csv");
        scale_file << "Threads,Time,Speedup\n";
        
        int thread_counts[] = {1, 2, 4, 8, 16};
        double base_time = 0.0;
        
        for (int t : thread_counts) {
            std::cout << "-> Ejecutando con " << t << " hilos... ";
            double t_total = 0;
            
            // Warm-up
            run_knn_evaluation(train_data, test_data, 5, samples, t, omp_sched_static, EUCLIDEAN, accuracy);
            
            for(int r = 0; r < num_runs; r++) {
                t_total += run_knn_evaluation(train_data, test_data, 5, samples, t, omp_sched_static, EUCLIDEAN, accuracy);
            }
            double t_avg = t_total / num_runs;
            
            if (t == 1) base_time = t_avg; // Tomamos el primer run como base de speedup (secuencial puro requeriria quitar pragmas, pero esto da una buena metrica)
            
            std::cout << t_avg << " seg | Speedup: " << base_time / t_avg << "x" << std::endl;
            scale_file << t << "," << t_avg << "," << base_time / t_avg << "\n";
        }
        scale_file.close();
        std::cout << "Pruebas de Escalabilidad terminadas. ('scaling_results.csv')" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Excepcion: " << e.what() << std::endl;
    }
    return 0;
}