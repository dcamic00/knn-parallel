#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

// Invertir bytes (Big-Endian a Little-Endian)
uint32_t swap_endian(uint32_t val) {
    return ((val << 24) & 0xff000000) |
           ((val <<  8) & 0x00ff0000) |
           ((val >>  8) & 0x0000ff00) |
           ((val >> 24) & 0x000000ff);
}

// Estructura SoA (Structure of Arrays)
struct DatasetSoA {
    std::vector<float> all_features; 
    std::vector<uint8_t> labels;     
    size_t num_samples;
    size_t num_features;
};

// Cargar imágenes
DatasetSoA read_mnist_images(const std::string& full_path) {
    DatasetSoA dataset;
    std::ifstream file(full_path, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0, num_images = 0, n_rows = 0, n_cols = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_endian(magic_number);
        if (magic_number != 2051) throw std::runtime_error("¡Archivo de imágenes inválido!");

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

        // Convertir a float y normalizar (0.0 a 1.0) para el cálculo de distancias
        for (size_t i = 0; i < temp_pixels.size(); ++i) {
            dataset.all_features[i] = static_cast<float>(temp_pixels[i]) / 255.0f;
        }
        std::cout << "-> Imágenes cargadas: " << num_images << " (" << dataset.num_features << " dimensiones)" << std::endl;
    } else {
        throw std::runtime_error("Error al abrir: " + full_path);
    }
    return dataset;
}

// Cargar etiquetas
void read_mnist_labels(const std::string& full_path, DatasetSoA& dataset) {
    std::ifstream file(full_path, std::ios::binary);
    if (file.is_open()) {
        uint32_t magic_number = 0, num_items = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = swap_endian(magic_number);
        if (magic_number != 2049) throw std::runtime_error("¡Archivo de etiquetas inválido!");

        file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
        num_items = swap_endian(num_items);
        if (num_items != dataset.num_samples) throw std::runtime_error("Desajuste entre imágenes y etiquetas.");

        dataset.labels.resize(num_items);
        file.read(reinterpret_cast<char*>(dataset.labels.data()), num_items);
        std::cout << "-> Etiquetas cargadas: " << num_items << std::endl;
    } else {
        throw std::runtime_error("Error al abrir: " + full_path);
    }
}

// -------------------------------------------------------------------------
// VERSIÓN SECUENCIAL
// -------------------------------------------------------------------------
float euclidean_distance_sq(const float* img1, const float* img2, size_t num_features) {
    float dist = 0.0f;
    for (size_t i = 0; i < num_features; ++i) {
        float diff = img1[i] - img2[i];
        dist += diff * diff;
    }
    return dist;
}

void evaluate_knn_sequential(const DatasetSoA& train, const DatasetSoA& test, int k, int max_test_samples = 1000) {
    std::cout << "\n[BASELINE] k-NN Secuencial (k=" << k << ") para " << max_test_samples << " muestras..." << std::endl;
    
    int correct_predictions = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < max_test_samples; ++i) {
        const float* current_test_img = &test.all_features[i * test.num_features];
        uint8_t true_label = test.labels[i];

        std::vector<std::pair<float, uint8_t>> distances(train.num_samples);

        for (size_t j = 0; j < train.num_samples; ++j) {
            const float* current_train_img = &train.all_features[j * train.num_features];
            float dist = euclidean_distance_sq(current_test_img, current_train_img, train.num_features);
            distances[j] = {dist, train.labels[j]};
        }

        std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                          [](const std::pair<float, uint8_t>& a, const std::pair<float, uint8_t>& b) {
                              return a.first < b.first;
                          });

        int votes[10] = {0};
        for (int v = 0; v < k; ++v) {
            votes[distances[v].second]++;
        }

        uint8_t predicted_label = 0;
        int max_votes = -1;
        for (int c = 0; c < 10; ++c) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                predicted_label = c;
            }
        }

        if (predicted_label == true_label) {
            correct_predictions++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    float accuracy = (static_cast<float>(correct_predictions) / max_test_samples) * 100.0f;
    std::cout << "-> Tiempo de ejecución: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "-> Precisión (Accuracy): " << accuracy << "%" << std::endl;
}


// -------------------------------------------------------------------------
// VERSIÓN PARALELA (OpenMP + SIMD)
// -------------------------------------------------------------------------
float euclidean_distance_sq_simd(const float* img1, const float* img2, size_t num_features) {
    float dist = 0.0f;
    // Forzar vectorización de los datos (procesar múltiples restas simultáneas)
    #pragma omp simd reduction(+:dist)
    for (size_t i = 0; i < num_features; ++i) {
        float diff = img1[i] - img2[i];
        dist += diff * diff;
    }
    return dist;
}

void evaluate_knn_parallel(const DatasetSoA& train, const DatasetSoA& test, int k, int max_test_samples = 1000) {
    std::cout << "\n[OPTIMIZADO] k-NN Paralelo (k=" << k << ") para " << max_test_samples << " muestras..." << std::endl;
    
    int correct_predictions = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Paralelizar el bucle externo distribuyendo el trabajo entre los hilos del procesador
    #pragma omp parallel for reduction(+:correct_predictions) schedule(dynamic)
    for (int i = 0; i < max_test_samples; ++i) {
        const float* current_test_img = &test.all_features[i * test.num_features];
        uint8_t true_label = test.labels[i];

        // Memoria local a cada hilo para evitar condiciones de carrera (Race Conditions)
        std::vector<std::pair<float, uint8_t>> distances(train.num_samples);

        for (size_t j = 0; j < train.num_samples; ++j) {
            const float* current_train_img = &train.all_features[j * train.num_features];
            float dist = euclidean_distance_sq_simd(current_test_img, current_train_img, train.num_features);
            distances[j] = {dist, train.labels[j]};
        }

        std::partial_sort(distances.begin(), distances.begin() + k, distances.end(),
                          [](const std::pair<float, uint8_t>& a, const std::pair<float, uint8_t>& b) {
                              return a.first < b.first;
                          });

        int votes[10] = {0};
        for (int v = 0; v < k; ++v) {
            votes[distances[v].second]++;
        }

        uint8_t predicted_label = 0;
        int max_votes = -1;
        for (int c = 0; c < 10; ++c) {
            if (votes[c] > max_votes) {
                max_votes = votes[c];
                predicted_label = c;
            }
        }

        if (predicted_label == true_label) {
            correct_predictions++;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    float accuracy = (static_cast<float>(correct_predictions) / max_test_samples) * 100.0f;
    std::cout << "-> Tiempo de ejecución: " << elapsed.count() << " segundos" << std::endl;
    std::cout << "-> Precisión (Accuracy): " << accuracy << "%" << std::endl;
}

int main() {
    std::string train_images_path = "train-images.idx3-ubyte";
    std::string train_labels_path = "train-labels.idx1-ubyte";
    std::string test_images_path = "t10k-images.idx3-ubyte";
    std::string test_labels_path = "t10k-labels.idx1-ubyte";
    
    try {
        std::cout << "--- Cargando Dataset de Entrenamiento ---" << std::endl;
        DatasetSoA train_data = read_mnist_images(train_images_path);
        read_mnist_labels(train_labels_path, train_data);

        std::cout << "\n--- Cargando Dataset de Prueba (Test) ---" << std::endl;
        DatasetSoA test_data = read_mnist_images(test_images_path);
        read_mnist_labels(test_labels_path, test_data);
        
        // Comparativa directa de rendimiento
        evaluate_knn_sequential(train_data, test_data, 5, 1000);
        evaluate_knn_parallel(train_data, test_data, 5, 1000);
        
    } catch (const std::exception& e) {
        std::cerr << "Excepción: " << e.what() << std::endl;
    }
    return 0;
}