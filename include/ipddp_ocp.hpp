#ifndef IPDDP_OCP_HPP
#define IPDDP_OCP_HPP

#include <symengine/expression.h>
#include <symengine/matrix.h>
#include <symengine/functions.h>
#include <symengine/symengine_exception.h>
#include <symengine/real_double.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <string>

#include "ipddp_options.hpp"

class IPDDP_OCP
{
public:
    int N;     // Prediction horizon
    int dim_x; // Dimension of states
    int dim_u; // Dimension of control inputs
    int dim_c; // Dimension of constraints
    double h;  // Timestep (sec)
    bool ws;   // Warm start

    SymEngine::DenseMatrix Q; // State penalty matrix in stage cost q
    SymEngine::DenseMatrix R; // Control penalty matrix in stage cost q
    SymEngine::DenseMatrix P; // State penalty matrix in terminal cost p
    SymEngine::DenseMatrix x; // State variables
    SymEngine::DenseMatrix u; // Control variables
    Eigen::MatrixXd x0_ws;
    Eigen::MatrixXd u0_ws;
    Eigen::MatrixXd x0;

    SymEngine::DenseMatrix f; // System dynamics
    SymEngine::DenseMatrix q; // Stage cost
    SymEngine::DenseMatrix p; // Terminal cost
    SymEngine::DenseMatrix c; // Constraints

    IPDDP_OCP() {}

    virtual ~IPDDP_OCP() {}

    virtual void initializeProblem() = 0;

private:
    void initializeParameters();
    void initializeMatrices();
    void initializeVariables();
    void defineDynamicsAndCosts();
};

#endif
