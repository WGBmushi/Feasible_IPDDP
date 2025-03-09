#ifndef IPDDP_UTILS_HPP
#define IPDDP_UTILS_HPP

#include <symengine/symbol.h>
#include <symengine/matrix.h>
#include <symengine/functions.h>
#include <symengine/expression.h>
#include <symengine/lambda_double.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <type_traits>
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <tuple>
#include <cmath>
#include <functional>

#include <fstream>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename T>
std::vector<SymEngine::DenseMatrix> HessianTensor(const SymEngine::DenseMatrix &fx, const SymEngine::DenseMatrix &x, T &fxx)
{
    int dim_x = x.nrows(); 

    for (int i = 0; i < dim_x; ++i)
    {
        SymEngine::DenseMatrix second_derivative(fx.nrows(), fx.ncols());
        SymEngine::RCP<const SymEngine::Basic> x_basic = x.get(i, 0);
        SymEngine::RCP<const SymEngine::Symbol> x_symbol = SymEngine::rcp_dynamic_cast<const SymEngine::Symbol>(x_basic);

        diff(fx, x_symbol, second_derivative);
        fxx[i] = second_derivative;
    }

    return fxx;
}

template <typename T>
Eigen::MatrixXd replaceAndEvaluate(
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &u,
    const SymEngine::DenseMatrix &funcs_x,
    const SymEngine::DenseMatrix &funcs_u,
    const T &symbolicMatrix)
{
    SymEngine::map_basic_basic subs_map;

    for (int i = 0; i < x.size(); ++i)
    {
        subs_map[funcs_x.get(i, 0)] = SymEngine::Expression(x[i]);
    }

    for (int i = 0; i < u.size(); ++i)
    {
        subs_map[funcs_u.get(i, 0)] = SymEngine::Expression(u[i]);
    }

    Eigen::MatrixXd result(symbolicMatrix.nrows(), symbolicMatrix.ncols());

    for (unsigned i = 0; i < symbolicMatrix.nrows(); ++i)
    {
        for (unsigned j = 0; j < symbolicMatrix.ncols(); ++j)
        {
            auto expr = symbolicMatrix.get(i, j)->subs(subs_map);
            result(i, j) = SymEngine::eval_double(*expr);
        }
    }

    return result;
}

template <typename T>
std::vector<Eigen::MatrixXd> replaceAndEvaluate(
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &u,
    const SymEngine::DenseMatrix &funcs_x,
    const SymEngine::DenseMatrix &funcs_u,
    const std::vector<T> &symbolicMatrices)
{
    std::vector<Eigen::MatrixXd> results;

    for (const auto &symbolicMatrix : symbolicMatrices)
    {
        Eigen::MatrixXd result = replaceAndEvaluate(x, u, funcs_x, funcs_u, symbolicMatrix);
        results.push_back(result);
    }

    return results;
}

template <typename T>
Eigen::MatrixXd tensdot(const Eigen::VectorXd &vec, const T &tensor)
{
    int vec_rows = vec.rows();
    int vec_cols = vec.cols();
    int tensor_rows = tensor[0].rows();
    int tensor_cols = tensor[0].cols();
    int dim_tensor = tensor.size();

    Eigen::MatrixXd result(tensor_cols, dim_tensor);

    for (int i = 0; i < dim_tensor; ++i)
    {
        Eigen::MatrixXd matrix = tensor[i];

        Eigen::MatrixXd vec_expanded;
        if (vec_rows == tensor_rows && vec_cols == 1)
        {
            vec_expanded = vec.replicate(1, tensor_cols);
        }
        else
        {
            std::cerr << "Dimensions of vec and tensor matrices do not match." << std::endl;
            exit(EXIT_FAILURE);
        }

        Eigen::MatrixXd dot_product = vec_expanded.cwiseProduct(matrix);
        Eigen::RowVectorXd col_sum = dot_product.colwise().sum();

        result.col(i) = col_sum.transpose();
    }

    return result;
}

template <typename T>
Eigen::MatrixXd tensdot(const T &tensor, const Eigen::VectorXd &vec)
{
    return tensdot(vec, tensor);
}

std::pair<Eigen::MatrixXd, int> chol(const Eigen::MatrixXd &matrix);

bool isInvertible(const Eigen::MatrixXd &mat);

// --------------------------------------------------------------------------------------------
// For debugging

void printHessianTensor(const std::vector<SymEngine::DenseMatrix> &HessianTensor);

template <typename T>
void printTensor(const std::vector<T> &tensor)
{
    for (const auto &mat : tensor)
    {
        std::cout << mat << std::endl;
        std::cout << "-------" << std::endl;
    }
}

// --------------------------------------------------------------------------------------------
// For evaluation

void plot_boxplot_from_file(const std::string &filename);

std::vector<double> read_column(const std::string &filename, int column_index);

void plot_box_plots(const std::string &filename,
                    const std::vector<double> &data1, const std::vector<double> &data2,
                    const std::vector<double> &data3, const std::vector<double> &data4,
                    const std::string &title,
                    const std::string &legend1, const std::string &legend2,
                    const std::string &legend3, const std::string &legend4);

void plot_box_plots(const std::string &filename,
                    const std::vector<double> &data1, const std::vector<double> &data2,
                    const std::vector<double> &data3, const std::vector<double> &data4,
                    const std::vector<double> &data5, const std::vector<double> &data6,
                    const std::vector<double> &data7, const std::vector<double> &data8,
                    const std::string &title,
                    const std::string &legend1, const std::string &legend2,
                    const std::string &legend3, const std::string &legend4,
                    const std::string &legend5, const std::string &legend6,
                    const std::string &legend7, const std::string &legend8);

#endif