#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

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

int main() {
    // Si tus archivos tienen nombres ligeramente distintos, cámbialos aquí
    std::string train_images_path = "train-images-idx3-ubyte";
    std::string train_labels_path = "train-labels-idx1-ubyte";
    std::string test_images_path = "t10k-images-idx3-ubyte";
    std::string test_labels_path = "t10k-labels-idx1-ubyte";
    
    try {
        std::cout << "--- Cargando Dataset de Entrenamiento ---" << std::endl;
        DatasetSoA train_data = read_mnist_images(train_images_path);
        read_mnist_labels(train_labels_path, train_data);

        std::cout << "\n--- Cargando Dataset de Prueba (Test) ---" << std::endl;
        DatasetSoA test_data = read_mnist_images(test_images_path);
        read_mnist_labels(test_labels_path, test_data);
        
        std::cout << "\n¡Todo listo para programar el k-NN!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Excepción: " << e.what() << std::endl;
    }
    return 0;
}