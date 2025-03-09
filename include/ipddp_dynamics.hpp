// dynamics_invpend.hpp

#ifndef IPDDP_DYNAMICS_HPP
#define IPDDP_DYNAMICS_HPP

#include <symengine/expression.h>
#include <symengine/derivative.h>
#include <symengine/matrix.h>
#include <symengine/printers.h>
#include <symengine/symbol.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <random>
#include <vector>
#include <cmath>

#include "ipddp_options.hpp"
#include "ipddp_ocp.hpp"

struct Functions
{
    SymEngine::DenseMatrix x;
    SymEngine::DenseMatrix u;
    SymEngine::DenseMatrix f;
    SymEngine::DenseMatrix fx;
    SymEngine::DenseMatrix fu;
    std::vector<SymEngine::DenseMatrix> fxx;
    std::vector<SymEngine::DenseMatrix> fxu;
    std::vector<SymEngine::DenseMatrix> fuu;

    SymEngine::DenseMatrix q;
    SymEngine::DenseMatrix qx;
    SymEngine::DenseMatrix qu;
    SymEngine::DenseMatrix qxx;
    SymEngine::DenseMatrix qxu;
    SymEngine::DenseMatrix quu;

    SymEngine::DenseMatrix p;
    SymEngine::DenseMatrix px;
    SymEngine::DenseMatrix pxx;

    SymEngine::DenseMatrix c;
    SymEngine::DenseMatrix cx;
    SymEngine::DenseMatrix cu;
    std::vector<SymEngine::DenseMatrix> cxx;
    std::vector<SymEngine::DenseMatrix> cxu;
    std::vector<SymEngine::DenseMatrix> cuu;

    int dim_c;
    int dim_x;
    int dim_u;
};

struct ForwardPass
{
    int horizon;
    int step;
    double cost;
    double logcost;
    double err;
    int failed;
    double stepsize;

    Eigen::MatrixXd x;
    Eigen::MatrixXd u;
    Eigen::MatrixXd y;
    Eigen::MatrixXd s;
    Eigen::MatrixXd mu;
    Eigen::VectorXd filter;
    Eigen::MatrixXd c;
    Eigen::MatrixXd q;

    Eigen::MatrixXd x_defect;
};

struct BackwardPass
{
    double reg;
    int failed;
    double recovery;
    double opterr;

    Eigen::MatrixXd ku;
    std::vector<Eigen::MatrixXd> Ku;
    Eigen::MatrixXd ky;
    std::vector<Eigen::MatrixXd> Ky;
    Eigen::MatrixXd ks;
    std::vector<Eigen::MatrixXd> Ks;
};

void dynamics(const IPDDP_OCP &ocp, Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg);

#endif // IPDDP_DYNAMICS_HPP
