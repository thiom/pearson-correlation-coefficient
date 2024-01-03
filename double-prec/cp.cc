#include <cmath>
#include <vector>

double row_mean(const float* row, int nx){
    double sum = 0.0, mean;
    for (int j=0; j < nx; j++){
        sum += (double) row[j];
    }
    mean = sum / (double) nx;
    return mean;
}


double row_root_sq_sum(const float* row, int nx, double mean){
    double square_sum = 0.0, s;
    for(int j=0; j < nx; j++){
        double val = (double) row[j];
        square_sum += ((val - mean) * (val - mean));
    }
    s = sqrt(square_sum);
    return s;
}

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> X;

    for (int row=0; row < ny; row++){
        double r_mean = row_mean(&data[row * nx], nx);
        double row_rss = row_root_sq_sum(&data[row * nx], nx, r_mean);
        for(int col=0; col < nx; col++){
            double val = (double) data[col + row * nx];
            double addval = (val - r_mean) / row_rss;
            X.push_back(addval);
        }
    }

    #pragma omp parallel for schedule(static, 1)
    for(int i=0; i < ny; i++){
        for(int j=i; j<ny; j++){
            double Y_ij = 0.0;
            for(int k=0; k < nx; k++){
                Y_ij += X[i * nx + k] * X[j * nx + k];
            }
            result[j + i * ny] = Y_ij;
        }
    }
}
