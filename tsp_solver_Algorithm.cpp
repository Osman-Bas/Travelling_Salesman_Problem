#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <numeric>
#include <omp.h>

using namespace std;

// Öklid mesafesi hesaplama
double euclidean_distance(const pair<double, double>& p1, const pair<double, double>& p2) {
    double dx = p1.first - p2.first;
    double dy = p1.second - p2.second;
    return sqrt(dx*dx + dy*dy);
}

// Mesafe matrisini önceden hesapla (OpenMP ile paralel)
vector<vector<double>> precompute_distances(const vector<pair<double, double>>& coords) {
    int n = coords.size();
    vector<vector<double>> dist_matrix(n, vector<double>(n));
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dist_matrix[i][j] = euclidean_distance(coords[i], coords[j]);
        }
    }
    return dist_matrix;
}

// Rota maliyeti hesapla
double calculate_route_distance(const vector<int>& route, const vector<vector<double>>& dist_matrix) {
    double total = 0.0;
    for (size_t i = 0; i < route.size() - 1; ++i) {
        total += dist_matrix[route[i]][route[i+1]];
    }
    total += dist_matrix[route.back()][route[0]];
    return total;
}

// 2-Opt optimizasyonu
vector<int> two_opt(vector<int> route, const vector<vector<double>>& dist_matrix) {
    bool improved = true;
    while (improved) {
        improved = false;
        for (size_t i = 1; i < route.size() - 2; ++i) {
            for (size_t j = i + 1; j < route.size(); ++j) {
                if (j - i == 1) continue;
                
                vector<int> new_route = route;
                reverse(new_route.begin() + i, new_route.begin() + j);
                
                double current_dist = calculate_route_distance(route, dist_matrix);
                double new_dist = calculate_route_distance(new_route, dist_matrix);
                
                if (new_dist < current_dist) {
                    route = new_route;
                    improved = true;
                }
            }
        }
    }
    return route;
}

// ILS ana algoritması (OpenMP ile paralel iterasyonlar)
pair<double, vector<int>> iterated_local_search(const vector<pair<double, double>>& coords, int max_iterations) {
    const auto dist_matrix = precompute_distances(coords);
    int n = coords.size();
    vector<int> best_route(n);
    iota(best_route.begin(), best_route.end(), 0);
    
    random_device rd;
    mt19937 gen(rd());
    shuffle(best_route.begin(), best_route.end(), gen);
    
    best_route = two_opt(best_route, dist_matrix);
    double best_cost = calculate_route_distance(best_route, dist_matrix);
    
    #pragma omp parallel for
    for (int iter = 0; iter < max_iterations; ++iter) {
        vector<int> new_route = best_route;
        
        // Pertürbasyon
        uniform_int_distribution<> dist(0, n-1);
        int i = dist(gen), j = dist(gen);
        if (i > j) swap(i, j);
        reverse(new_route.begin() + i, new_route.begin() + j);
        
        new_route = two_opt(new_route, dist_matrix);
        double new_cost = calculate_route_distance(new_route, dist_matrix);
        
        #pragma omp critical
        {
            if (new_cost < best_cost) {
                best_route = new_route;
                best_cost = new_cost;
            }
        }
        
        if (iter % 100 == 0) {
            #pragma omp critical
            {
                cout << "Iteration: " << iter << "/" << max_iterations
                     << " Best Cost: " << best_cost << endl;
            }
        }
    }
    
    return {best_cost, best_route};
}

// Veri okuma fonksiyonu
pair<int, vector<pair<double, double>>> read_coordinates(const string& file_path) {
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Dosya açılamadı: " << file_path << endl;
        exit(1);
    }
    
    int num_cities;
    file >> num_cities;
    
    vector<pair<double, double>> coords;
    double x, y;
    while (file >> x >> y) {
        coords.emplace_back(x, y);
    }
    
    return {num_cities, coords};
}

// Sonuç yazdırma
void write_output(const string& file_path, double cost, const vector<int>& path) {
    ofstream file(file_path);
    file.precision(2);
    file << fixed;
    file << "Optimal maliyet değeri: " << cost << "\n";
    file << "Optimal rota: ";
    for (int city : path) file << city << " ";
    file << "\n";
}

int main() {
    // OpenMP ayarları
    omp_set_num_threads(8); // M4 Pro için 8 çekirdek
    cout << "OpenMP aktif. Kullanılan çekirdek sayısı: " << omp_get_max_threads() << endl;

    // Dosya yolları
    const string input_file = "/Users/osman/Desktop/YAZILIM/VS_Code/tsp_3038_1.txt";
    const string output_file = "/Users/osman/Desktop/YAZILIM/VS_Code/tsp_result_cpp_optimized.txt";
    
    // Veriyi oku
    auto [num_cities, coords] = read_coordinates(input_file);
    cout << num_cities << " şehir yüklendi. ILS başlıyor..." << endl;
    
    // Zaman ölçümü
    auto start = chrono::high_resolution_clock::now();
    
    // ILS çalıştır
    auto [best_cost, best_route] = iterated_local_search(coords, 1000);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    
    // Sonuçları yaz
    write_output(output_file, best_cost, best_route);
    
    cout << "Tamamlandı! Toplam süre: " << elapsed.count() << " saniye\n";
    cout << "En iyi maliyet: " << best_cost << "\n";
    cout << "Sonuçlar '" << output_file << "' dosyasına kaydedildi.\n";
    
    return 0;
}
