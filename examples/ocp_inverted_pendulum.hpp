#ifndef OCP_FORMULATION_INVP_HPP
#define OCP_FORMULATION_INVP_HPP

#include "ipddp_ocp.hpp"
#include "ipddp_options.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class INVP_OCP : public IPDDP_OCP
{
public:
    INVP_OCP()
    {
        initializeProblem();
    }

    void initializeProblem() override
    {
        // Initialization specific to INVP_OCP
        initializeParameters();
        initializeMatrices();
        initializeVariables();
        defineDynamicsAndCosts();
    }

private:
    bool index_ws = false;

    void initializeParameters()
    {
        // Dimensions and constants
        N = 500;
        dim_x = 2;
        dim_u = 1;
        dim_c = 2;
        h = 0.05;
        ws = index_ws;
    }

    void initializeMatrices()
    {
        Q = SymEngine::DenseMatrix(dim_x, dim_x);
        SymEngine::eye(Q);
        R = SymEngine::DenseMatrix(dim_u, dim_u);
        R.set(0, 0, SymEngine::Expression(1));
        P = SymEngine::DenseMatrix(dim_x, dim_x);
        SymEngine::zeros(P);
        P.set(0, 0, SymEngine::Expression(10));
        P.set(1, 1, SymEngine::Expression(10));
    }

    void initializeVariables()
    {
        x = SymEngine::DenseMatrix(dim_x, 1);
        u = SymEngine::DenseMatrix(dim_u, 1);
        x0 = Eigen::MatrixXd(dim_x, 1);

        for (int i = 0; i < dim_x; ++i)
        {
            std::string var_name = "x" + std::to_string(i + 1);
            x.set(i, 0, SymEngine::symbol(var_name));
        }

        for (int i = 0; i < dim_u; ++i)
        {
            std::string var_name = "u" + std::to_string(i + 1);
            u.set(i, 0, SymEngine::symbol(var_name));
        }

        x0 << -M_PI,
            0;
    }

    void defineDynamicsAndCosts()
    {
        SymEngine::DenseMatrix x_vec(dim_x, 1);

        for (int i = 0; i < dim_x; ++i)
        {
            x_vec.set(i, 0, x.get(i, 0));
        }

        SymEngine::DenseMatrix u_vec(dim_u, 1);
        // u_vec.set(0, 0, u.get(0, 0));
        for (int i = 0; i < dim_x; ++i)
        {
            u_vec.set(i, 0, u.get(i, 0));
        }

        // Define dynamics
        f = SymEngine::DenseMatrix(dim_x, 1);
        SymEngine::Expression h_expr = SymEngine::Expression(h);
        f.set(0, 0, x.get(0, 0) + h_expr * x.get(1, 0));
        f.set(1, 0, x.get(1, 0) + h_expr * SymEngine::sin(x.get(0, 0)) + h_expr * u.get(0, 0));

        // Define cost functions
        SymEngine::DenseMatrix x_cost(dim_x, 1), u_cost(dim_u, 1);
        mul_dense_dense(Q, x_vec, x_cost);

        mul_dense_dense(R, u_vec, u_cost);

        // Transpose state and control vectors
        SymEngine::DenseMatrix x_trans(1, dim_x), u_trans(1, dim_u);
        transpose_dense(x_vec, x_trans);

        transpose_dense(u_vec, u_trans);

        // Calculate quadratic cost terms
        SymEngine::DenseMatrix q1_x(1, 1), q1_u(1, 1);
        SymEngine::dot(x_trans, x_cost, q1_x);
        SymEngine::dot(u_trans, u_cost, q1_u);

        // Modify q to be a DenseMatrix
        q = SymEngine::DenseMatrix(1, 1);
        q.set(0, 0, 0.5 * h_expr * SymEngine::add(q1_x.get(0, 0), q1_u.get(0, 0)));

        // Calculate linear cost term
        SymEngine::DenseMatrix p_cost(dim_x, 1);
        mul_dense_dense(P, x_vec, p_cost);

        SymEngine::DenseMatrix p1_x(1, 1);
        SymEngine::dot(x_trans, p_cost, p1_x);

        // Modify p to be a DenseMatrix
        p = SymEngine::DenseMatrix(1, 1);
        SymEngine::Expression p_c1 = SymEngine::Expression(0.5);
        p.set(0, 0, p_c1 * p1_x.get(0, 0));

        // Define constraints
        SymEngine::Expression c_c1 = SymEngine::Expression(0.25);
        c = SymEngine::DenseMatrix(dim_c, 1);
        c.set(0, 0, u.get(0, 0) - c_c1);
        c.set(1, 0, SymEngine::neg(u.get(0, 0)) - c_c1);
    }
};

void plot_phase_trajectory(const Eigen::MatrixXd &state)
{
    Eigen::VectorXd x0 = state.row(0);
    Eigen::VectorXd x1 = state.row(1);
    std::vector<double> x_data(x0.data(), x0.data() + x0.size());
    std::vector<double> y_data(x1.data(), x1.data() + x1.size());
    plt::plot(x_data, y_data);
    plt::xlabel("x0");
    plt::ylabel("x1");
    plt::title("Phase trajectory");
    plt::show();
}

void plot_control_input(const Eigen::MatrixXd &control)
{
    Eigen::VectorXd u = control.row(0);
    std::vector<double> u_data(u.data(), u.data() + u.size());
    plt::plot(u_data);
    plt::xlabel("N");
    plt::ylabel("u");
    plt::title("Control input");
    plt::show();
}

void plot_cost(const std::vector<double> &cost)
{
    plt::plot(cost);
    plt::xlabel("N");
    plt::ylabel("cost");
    plt::title("Cost");
    plt::show();
}

#endif
