#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

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

float euclidean_distance_sq_simd(const float* img1, const float* img2, size_t num_features) {
    float dist = 0.0f;
    #pragma omp simd reduction(+:dist)
    for (size_t i = 0; i < num_features; ++i) {
        float diff = img1[i] - img2[i];
        dist += diff * diff;
    }
    return dist;
}



// 1. DYNAMIC SCHEDULING



double test_dynamic(const DatasetSoA& train, const DatasetSoA& test, int k, int samples, int threads) {
    omp_set_num_threads(threads);
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:correct) schedule(dynamic)
    for (int i = 0; i < samples; ++i) {
        const float* current_test_img = &test.all_features[i * test.num_features];
        std::vector<std::pair<float, uint8_t>> distances(train.num_samples);

        for (size_t j = 0; j < train.num_samples; ++j) {
            float dist = euclidean_distance_sq_simd(current_test_img, &train.all_features[j * train.num_features], train.num_features);
            distances[j] = {dist, train.labels[j]};
        }
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                          [](const std::pair<float, uint8_t>& a, const std::pair<float, uint8_t>& b) { return a.first < b.first; });
    }
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    return elapsed.count();
}



// 2. STATIC SCHEDULING



double test_static(const DatasetSoA& train, const DatasetSoA& test, int k, int samples, int threads) {
    omp_set_num_threads(threads);
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int i = 0; i < samples; ++i) {
        const float* current_test_img = &test.all_features[i * test.num_features];
        std::vector<std::pair<float, uint8_t>> distances(train.num_samples);

        for (size_t j = 0; j < train.num_samples; ++j) {
            float dist = euclidean_distance_sq_simd(current_test_img, &train.all_features[j * train.num_features], train.num_features);
            distances[j] = {dist, train.labels[j]};
        }
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                          [](const std::pair<float, uint8_t>& a, const std::pair<float, uint8_t>& b) { return a.first < b.first; });
    }
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
    return elapsed.count();
}




// 3. GUIDED SCHEDULING



double test_guided(const DatasetSoA& train, const DatasetSoA& test, int k, int samples, int threads) {
    omp_set_num_threads(threads);
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:correct) schedule(guided)
    for (int i = 0; i < samples; ++i) {
        const float* current_test_img = &test.all_features[i * test.num_features];
        std::vector<std::pair<float, uint8_t>> distances(train.num_samples);

        for (size_t j = 0; j < train.num_samples; ++j) {
            float dist = euclidean_distance_sq_simd(current_test_img, &train.all_features[j * train.num_features], train.num_features);
            distances[j] = {dist, train.labels[j]};
        }
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                          [](const std::pair<float, uint8_t>& a, const std::pair<float, uint8_t>& b) { return a.first < b.first; });
    }
    std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start;
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
        int k = 5;
        int optimal_threads = 8; // Fijado segun tu mejor resultado
        
        std::ofstream csv_file("scheduling_results.csv");
        csv_file << "Schedule_Type,Time_Seconds\n";

        std::cout << "\nIniciando comparativa de Scheduling OpenMP con " << optimal_threads << " hilos..." << std::endl;
        
        // 1. Dynamic
        std::cout << "-> Probando schedule(dynamic)... ";
        double t_dyn = 0;
        for(int r=0; r<3; r++) t_dyn += test_dynamic(train_data, test_data, k, samples, optimal_threads);
        t_dyn /= 3.0;
        std::cout << t_dyn << " seg" << std::endl;
        csv_file << "dynamic," << t_dyn << "\n";

        // 2. Static
        std::cout << "-> Probando schedule(static)... ";
        double t_stat = 0;
        for(int r=0; r<3; r++) t_stat += test_static(train_data, test_data, k, samples, optimal_threads);
        t_stat /= 3.0;
        std::cout << t_stat << " seg" << std::endl;
        csv_file << "static," << t_stat << "\n";

        // 3. Guided
        std::cout << "-> Probando schedule(guided)... ";
        double t_gui = 0;
        for(int r=0; r<3; r++) t_gui += test_guided(train_data, test_data, k, samples, optimal_threads);
        t_gui /= 3.0;
        std::cout << t_gui << " seg" << std::endl;
        csv_file << "guided," << t_gui << "\n";
        
        csv_file.close();
        std::cout << "\nPruebas de Scheduling terminadas. Guardado en 'scheduling_results.csv'" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Excepcion: " << e.what() << std::endl;
    }
    return 0;
}