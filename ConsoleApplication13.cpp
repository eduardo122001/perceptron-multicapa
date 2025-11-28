#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

using namespace std;

// Optimizado: se eliminaron threads por neurona, se reservan vectores auxiliares,
// se reducen prints en el loop y se activan optimizaciones de I/O.

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float x) {
    float f = sigmoid(x);
    return f * (1 - f);
}

struct Image {
    int label;                       // la clase (0–9)
    vector<float> r;
    vector<float> g;
    vector<float> b;
};

struct neuron {
    vector<float> weights;
    float bias;
    float lr;
    float u;
    float dy;
    float e{ 0 };

    neuron(int cantentrada) {
        lr = 0.1f;
        bias = 1;
        weights.resize(cantentrada + 1);
        for (auto& w : weights)
            w = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float calculate(const vector<float>& caca) {
        float summatory = bias * weights[0];
        int n = (int)caca.size();
        for (int x = 0; x < n; ++x)
            summatory += caca[x] * weights[x + 1];
        u = summatory;
        dy = sigmoid(u);
        return dy;
    }

    float calculate_entry(const Image& objective) {
        float summatory = bias * weights[0];

        const float* rptr = objective.r.data();
        const float* gptr = objective.g.data();
        const float* bptr = objective.b.data();

        for (int x = 0; x < 1024; ++x) {
            summatory += rptr[x] * weights[x + 1];
            summatory += gptr[x] * weights[x + 1024 + 1];
            summatory += bptr[x] * weights[x + 2048 + 1];
        }
        u = summatory;
        dy = sigmoid(u);
        return dy;
    }
};

struct finallayer {
    vector<neuron> neuronas;

    finallayer(int cantidadneu, int cantpes) {
        neuronas.reserve(cantidadneu);
        for (int i = 0; i < cantidadneu; i++)
            neuronas.emplace_back(cantpes);
    }

    vector<float> calculate_neurons(const vector<float>& obj) {
        vector<float> result(neuronas.size());
        for (size_t i = 0; i < neuronas.size(); ++i)
            result[i] = neuronas[i].calculate(obj);
        return result;
    }

    void update_error(int label) {
        for (int xn = 0; xn < (int)neuronas.size(); xn++) {
            if (xn == label)
                neuronas[xn].e = (1 - neuronas[xn].dy) * sigmoid_derivative(neuronas[xn].u);
            else
                neuronas[xn].e = (0 - neuronas[xn].dy) * sigmoid_derivative(neuronas[xn].u);
        }
    }

    void update_weight(const vector<float>& obj) {
        for (int xn = 0; xn < (int)neuronas.size(); xn++) {
            neuronas[xn].weights[0] += neuronas[xn].lr * neuronas[xn].bias * neuronas[xn].e;

            for (int n = 1; n < (int)neuronas[xn].weights.size(); n++)
                neuronas[xn].weights[n] += neuronas[xn].lr * obj[n - 1] * neuronas[xn].e;
        }
    }

    void update(int obj, const vector<float>& obj2) {
        update_error(obj);
        update_weight(obj2);
    }
};

struct normalhidenlayer {
    vector<neuron> neuronas;

    normalhidenlayer(int cantidadneu, int cantpes) {
        neuronas.reserve(cantidadneu);
        for (int i = 0; i < cantidadneu; i++)
            neuronas.emplace_back(cantpes);
    }

    vector<float> calculate_neurons(const vector<float>& obj) {
        vector<float> result(neuronas.size());
        for (size_t i = 0; i < neuronas.size(); ++i)
            result[i] = neuronas[i].calculate(obj);
        return result;
    }

    void update_error(const finallayer& obj) {
        float sum{ 0 };
        for (int l = 0; l < (int)neuronas.size(); l++) {
            for (int xn = 0; xn < (int)obj.neuronas.size(); xn++)
                sum += obj.neuronas[xn].weights[l + 1] * obj.neuronas[xn].e;

            neuronas[l].e = sigmoid_derivative(neuronas[l].u) * sum;
            sum = 0;
        }
    }

    void update_weight(const vector<float>& obj) {
        for (int xn = 0; xn < (int)neuronas.size(); xn++) {
            neuronas[xn].weights[0] += neuronas[xn].lr * neuronas[xn].bias * neuronas[xn].e;

            for (int n = 1; n < (int)neuronas[xn].weights.size(); n++)
                neuronas[xn].weights[n] += neuronas[xn].lr * obj[n - 1] * neuronas[xn].e;
        }
    }

    void update(const finallayer& obj, const vector<float>& obj2) {
        update_error(obj);
        update_weight(obj2);
    }
};

struct inputhidenlayer {
    vector<neuron> neuronas;

    inputhidenlayer(int cantidadneu, int cantpes) {
        neuronas.reserve(cantidadneu);
        for (int i = 0; i < cantidadneu; i++)
            neuronas.emplace_back(cantpes);
    }

    vector<float> calculate_neurons(const Image& obj) {
        vector<float> result(neuronas.size());
        for (size_t i = 0; i < neuronas.size(); ++i)
            result[i] = neuronas[i].calculate_entry(obj);
        return result;
    }

    void update_error(const normalhidenlayer& obj) {
        float sum{ 0 };
        for (int l = 0; l < (int)neuronas.size(); l++) {
            for (int xn = 0; xn < (int)obj.neuronas.size(); xn++)
                sum += obj.neuronas[xn].weights[l + 1] * obj.neuronas[xn].e;

            neuronas[l].e = sigmoid_derivative(neuronas[l].u) * sum;
            sum = 0;
        }
    }

    void update_weight(const Image& obj) {
        for (int xn = 0; xn < (int)neuronas.size(); xn++) {
            neuronas[xn].weights[0] += neuronas[xn].lr * neuronas[xn].bias * neuronas[xn].e;

            for (int n = 0; n < 1024; ++n) {
                neuronas[xn].weights[n + 1] += neuronas[xn].lr * obj.r[n] * neuronas[xn].e;
                neuronas[xn].weights[n + 1 + 1024] += neuronas[xn].lr * obj.g[n] * neuronas[xn].e;
                neuronas[xn].weights[n + 1 + 2048] += neuronas[xn].lr * obj.b[n] * neuronas[xn].e;
            }
        }
    }

    void update(const normalhidenlayer& obj, const Image& obj2) {
        update_error(obj);
        update_weight(obj2);
    }
};

struct MLP {
    inputhidenlayer L1;
    normalhidenlayer L2;
    finallayer L3;

    vector<float> aux1;
    vector<float> aux2;
    vector<float> aux3;

    MLP(int entradas, int h1, int h2, int out)
        : L1(h1, entradas), L2(h2, h1), L3(out, h2) {
        aux1.reserve(h1);
        aux2.reserve(h2);
        aux3.reserve(out);
    }

    vector<float> calculate(const Image& data) {
        aux1 = L1.calculate_neurons(data);
        aux2 = L2.calculate_neurons(aux1);
        aux3 = L3.calculate_neurons(aux2);
        return aux3;
    }

    bool verify(const Image& obj) {
        int pred = max_element(aux3.begin(), aux3.end()) - aux3.begin();
        return pred == obj.label;
    }

    void train(const Image& obj) {
        L3.update(obj.label, aux2);
        L2.update(L3, aux1);
        L1.update(L2, obj);
    }
};

Image readSingleCIFARImage(ifstream& file) {
    Image img;
    unsigned char label;

    file.read((char*)&label, 1);
    if (!file) throw runtime_error("Fin archivo");
    img.label = label;

    const int size = 32 * 32;
    vector<unsigned char> r_bytes(size), g_bytes(size), b_bytes(size);

    file.read((char*)r_bytes.data(), size);
    file.read((char*)g_bytes.data(), size);
    file.read((char*)b_bytes.data(), size);

    img.r.resize(size);
    img.g.resize(size);
    img.b.resize(size);

    for (int i = 0; i < size; ++i) {
        img.r[i] = (r_bytes[i] / 255.0f - 0.5f) * 2.0f;
        img.g[i] = (g_bytes[i] / 255.0f - 0.5f) * 2.0f;
        img.b[i] = (b_bytes[i] / 255.0f - 0.5f) * 2.0f;
    }
    return img;
}

vector<Image> loadCIFAR10File(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open())
        throw runtime_error("No se pudo abrir " + filename);

    vector<Image> images;
    images.reserve(10000);
    for (int i = 0; i < 10000; ++i)
        images.push_back(readSingleCIFARImage(file));

    return images;
}

vector<Image> loadAllCIFAR10Train(const string& base_path) {
    vector<Image> dataset;
    dataset.reserve(50000);
    for (int i = 1; i <= 5; ++i) {
        string filename = base_path + "/data_batch_" + to_string(i) + ".bin";
        cout << "Leyendo " << filename << " ..." << endl;
        auto batch = loadCIFAR10File(filename);
        dataset.insert(dataset.end(), batch.begin(), batch.end());
    }
    cout << "Total cargadas: " << dataset.size() << endl;
    return dataset;
}

// === Guardado y carga de pesos ===

template<typename MLPType>
void save_model(const MLPType& model, const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out) {
        cout << "No se pudo guardar el modelo\n";
        return;
    }

    auto save_layer = [&](const auto& layer) {
        for (const auto& neuron : layer.neuronas) {
            out.write(reinterpret_cast<const char*>(neuron.weights.data()), neuron.weights.size() * sizeof(float));
            out.write(reinterpret_cast<const char*>(&neuron.bias), sizeof(float));
        }
        };

    save_layer(model.L1);
    save_layer(model.L2);
    save_layer(model.L3);
}

template<typename MLPType>
void load_model(MLPType& model, const string& filename) {
    ifstream in(filename, ios::binary);
    if (!in) {
        cout << "Modelo no encontrado, se entrenara desde cero.\n";
        return;
    }

    auto load_layer = [&](auto& layer) {
        for (auto& neuron : layer.neuronas) {
            in.read(reinterpret_cast<char*>(neuron.weights.data()), neuron.weights.size() * sizeof(float));
            in.read(reinterpret_cast<char*>(&neuron.bias), sizeof(float));
        }
        };

    load_layer(model.L1);
    load_layer(model.L2);
    load_layer(model.L3);

    cout << "Modelo cargado correctamente.\n";
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    srand((unsigned)time(nullptr));

    vector<Image> train_data = loadAllCIFAR10Train("C:/Users/Usuario/Desktop/perce/cifar-10-batches-bin");

    MLP mlp(3072, 6, 4, 10);

    // === CARGAR MODELO SI EXISTE ===
    load_model(mlp, "modelo.bin");

    // === ENTRENAMIENTO ===
    for (int epoch = 0; epoch < 20; ++epoch) {
        cout << "\nEpoca " << epoch + 1 << "...";
        int correct = 0;

        for (auto& img : train_data) {
            mlp.calculate(img);
            correct += mlp.verify(img);
            mlp.train(img);
        }

        cout << " precision -> "
            << (float)correct / (float)train_data.size() * 100.0f << "%" << endl;
    }

    
    save_model(mlp, "modelo.bin");
    cout << "\nModelo guardado en 'modelo.bin'.\n";

    
    vector<Image> test_data =
        loadCIFAR10File("C:/Users/Usuario/Desktop/perce/cifar-10-batches-bin/test_batch.bin");

    cout << "\nEvaluando en test_batch..." << endl;

    int correct_test = 0;

    for (auto& img : test_data) {
        auto out = mlp.calculate(img);
        int pred = max_element(out.begin(), out.end()) - out.begin();

        if (pred == img.label)
            correct_test++;
    }

    cout << "Precisión en test: "
        << (float)correct_test / test_data.size() * 100.0f
        << "%" << endl;

    return 0;
}
