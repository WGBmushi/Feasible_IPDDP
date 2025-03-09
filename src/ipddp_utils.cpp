#include "ipddp_utils.hpp"

std::pair<Eigen::MatrixXd, int> chol(const Eigen::MatrixXd &matrix)
{
    Eigen::LLT<Eigen::MatrixXd> llt(matrix);
    if (llt.info() == Eigen::Success)
    {
        return std::make_pair(llt.matrixL(), 0);
    }
    else
    {
        return std::make_pair(Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols()), 1);
    }
}

bool isInvertible(const Eigen::MatrixXd &mat)
{
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(mat);
    return cod.rank() == mat.cols();
}

// --------------------------------------------------------------------------------------------
// For debugging

void printHessianTensor(const std::vector<SymEngine::DenseMatrix> &HessianTensor)
{
    int index = 0; 
    for (const auto &hessian : HessianTensor)
    {

        std::cout << hessian << std::endl;
        std::cout << "-------" << std::endl;
        index++; 
    }
}


// --------------------------------------------------------------------------------------------
// For evaluation

void plot_boxplot_from_file(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    std::vector<double> iteration_steps;
    std::vector<double> iteration_times;
    std::vector<double> perturbation_params;
    std::vector<double> loss_values;
    std::vector<double> total_times;

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        double iteration_step, iteration_time, perturbation_param, loss_value, total_time;
        ss >> iteration_step >> iteration_time >> perturbation_param >> loss_value >> total_time;

        iteration_steps.push_back(iteration_step);
        iteration_times.push_back(iteration_time);
        perturbation_params.push_back(perturbation_param);
        loss_values.push_back(loss_value);
        total_times.push_back(total_time);
    }
    file.close();

    if (iteration_steps.empty())
    {
        std::cerr << "No data to plot." << std::endl;
        return;
    }

    std::vector<std::vector<double>> data = {
        iteration_steps,
        iteration_times,
        perturbation_params,
        loss_values,
        total_times};

    plt::figure_size(800, 600);
    plt::boxplot(data);
    plt::title("Box plot of optimization process data");
    plt::ylabel("Values");
    plt::show();
}

// Function to read a column from a file and return it as a vector
std::vector<double> read_column(const std::string &filename, int column_index)
{
    std::vector<double> column_data;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Could not open the file " << filename << std::endl;
        return column_data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;
        int current_index = 0;
        while (std::getline(ss, value, ' '))
        { // Assuming space as the delimiter
            if (current_index == column_index)
            {
                column_data.push_back(std::stod(value));
                break;
            }
            current_index++;
        }
    }

    file.close();
    return column_data;
}

void plot_box_plots(const std::string &filename,
                    const std::vector<double> &data1, const std::vector<double> &data2,
                    const std::vector<double> &data3, const std::vector<double> &data4,
                    const std::string &title,
                    const std::string &legend1, const std::string &legend2,
                    const std::string &legend3, const std::string &legend4)
{
    std::vector<double> ticks = {1.0, 2.0, 3.0, 4.0};
    std::vector<std::string> labels = {legend1, legend3, legend2, legend4};
    std::map<std::string, std::string> keywords = {};

    plt::figure();
    plt::boxplot(std::vector<std::vector<double>>{data1, data3, data2, data4});
    plt::title(title);
    plt::xticks(ticks, labels, keywords);
    plt::xlabel("Algorithm");
    plt::ylabel("Elapsed time (sec)");
    // plt::show();
    std::cout << "Saved plot to " << filename << std::endl; // Debug information
    plt::save(filename);
}

void plot_box_plots(const std::string &filename,
                    const std::vector<double> &data1, const std::vector<double> &data2,
                    const std::vector<double> &data3, const std::vector<double> &data4,
                    const std::vector<double> &data5, const std::vector<double> &data6,
                    const std::vector<double> &data7, const std::vector<double> &data8,
                    const std::string &title,
                    const std::string &legend1, const std::string &legend2,
                    const std::string &legend3, const std::string &legend4,
                    const std::string &legend5, const std::string &legend6,
                    const std::string &legend7, const std::string &legend8)
{
    std::vector<double> ticks = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<std::string> labels = {legend1, legend3, legend2, legend4, legend5, legend7, legend6, legend8};
    std::map<std::string, std::string> keywords = {};

    plt::figure();
    plt::boxplot(std::vector<std::vector<double>>{data1, data3, data2, data4, data5, data7, data6, data8});
    plt::title(title);
    plt::xticks(ticks, labels, keywords);
    plt::xlabel("Algorithm");
    plt::ylabel("Elapsed time (sec)");
    // plt::show();
    std::cout << "Saved plot to " << filename << std::endl; // Debug information
    plt::save(filename);
}
