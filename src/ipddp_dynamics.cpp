
#include "ipddp_base.hpp"
#include <omp.h>

void dynamics(const IPDDP_OCP &ocp, Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg)
{
    int N = ocp.N;
    int dim_x = ocp.dim_x;
    int dim_u = ocp.dim_u;
    int dim_c = ocp.dim_c;
    auto x = ocp.x;
    auto u = ocp.u;
    auto f = ocp.f;
    auto q = ocp.q;
    auto p = ocp.p;
    auto c = ocp.c;
    alg.ws = ocp.ws;

#pragma omp parallel sections
    {
#pragma omp section
        {
            funcs.x = SymEngine::DenseMatrix(dim_x, 1);
            funcs.x = x;

            funcs.u = SymEngine::DenseMatrix(dim_u, 1);
            funcs.u = u;

            funcs.f = SymEngine::DenseMatrix(dim_x, 1);
            funcs.f = f;

            funcs.fx = SymEngine::DenseMatrix(dim_x, dim_x);
            jacobian(f, x, funcs.fx);

            funcs.fu = SymEngine::DenseMatrix(dim_x, dim_u);
            jacobian(f, u, funcs.fu);

            funcs.fxx = std::vector<SymEngine::DenseMatrix>(dim_x);
            HessianTensor(funcs.fx, x, funcs.fxx);

            funcs.fxu = std::vector<SymEngine::DenseMatrix>(dim_u);
            HessianTensor(funcs.fx, u, funcs.fxu);

            funcs.fuu = std::vector<SymEngine::DenseMatrix>(dim_u);
            HessianTensor(funcs.fu, u, funcs.fuu);

            funcs.q = SymEngine::DenseMatrix(1, 1);
            funcs.q = q;

            funcs.qx = SymEngine::DenseMatrix(dim_x, 1);
            SymEngine::DenseMatrix qx_temp(1, dim_x);
            jacobian(q, x, qx_temp);
            transpose_dense(qx_temp, funcs.qx);

            funcs.qu = SymEngine::DenseMatrix(dim_u, 1);
            SymEngine::DenseMatrix qu_temp(1, dim_u);
            jacobian(q, u, qu_temp);
            transpose_dense(qu_temp, funcs.qu);

            funcs.qxx = SymEngine::DenseMatrix(dim_x, dim_x);
            jacobian(funcs.qx, x, funcs.qxx);

            funcs.qxu = SymEngine::DenseMatrix(dim_x, dim_u);
            jacobian(funcs.qx, u, funcs.qxu);

            funcs.quu = SymEngine::DenseMatrix(dim_u, dim_u);
            jacobian(funcs.qu, u, funcs.quu);

            funcs.p = SymEngine::DenseMatrix(1, 1);
            funcs.p = p;

            funcs.px = SymEngine::DenseMatrix(dim_x, 1);
            SymEngine::DenseMatrix px_temp(1, dim_x);
            jacobian(p, x, px_temp);
            transpose_dense(px_temp, funcs.px);

            funcs.pxx = SymEngine::DenseMatrix(dim_x, dim_x);
            jacobian(funcs.px, x, funcs.pxx);

            // ---------------------------------------------------------
            if (!dim_c)
            {
                funcs.c = SymEngine::DenseMatrix();
                funcs.cx = SymEngine::DenseMatrix();
                funcs.cu = SymEngine::DenseMatrix();
                funcs.cxx = std::vector<SymEngine::DenseMatrix>();
                funcs.cxu = std::vector<SymEngine::DenseMatrix>();
                funcs.cuu = std::vector<SymEngine::DenseMatrix>();
            }
            else
            {
                funcs.c = SymEngine::DenseMatrix(dim_c, 1);
                funcs.c = c;

                funcs.cx = SymEngine::DenseMatrix(dim_c, dim_x);
                jacobian(c, x, funcs.cx);

                funcs.cu = SymEngine::DenseMatrix(dim_c, dim_u);
                jacobian(c, u, funcs.cu);

                funcs.cxx = std::vector<SymEngine::DenseMatrix>(dim_x);
                HessianTensor(funcs.cx, x, funcs.cxx);

                funcs.cxu = std::vector<SymEngine::DenseMatrix>(dim_u);
                HessianTensor(funcs.cx, u, funcs.cxu);

                funcs.cuu = std::vector<SymEngine::DenseMatrix>(dim_u);
                HessianTensor(funcs.cu, u, funcs.cuu);
            }

            funcs.dim_c = dim_c;
            funcs.dim_x = dim_x;
            funcs.dim_u = dim_u;
        }

#pragma omp section
        {
            fp.horizon = N;

            if (alg.ws)
            {
                fp.x = ocp.x0_ws;
                fp.u = ocp.u0_ws;
            }
            else
            {
                fp.x = Eigen::MatrixXd::Zero(dim_x, N + 1);
                fp.x.col(0) = ocp.x0;
                std::default_random_engine generator;
                std::uniform_real_distribution<double> distribution(-0.01, 0.01);

                fp.u = Eigen::MatrixXd::NullaryExpr(dim_u, N, [&]()
                                                    { return distribution(generator); });
            }

            fp.y = Eigen::MatrixXd::Constant(dim_c, N, 0.01);
            fp.s = Eigen::MatrixXd::Constant(dim_c, N, 0.1);
            fp.mu = fp.y.cwiseProduct(fp.s);
            fp.filter = Eigen::MatrixXd(2, 1);
            fp.filter << std::numeric_limits<double>::infinity(), 0;
            fp.c = Eigen::MatrixXd::Zero(dim_c, N);
            fp.q = Eigen::MatrixXd::Zero(1, N);
            fp.cost = 0.0;
            fp.logcost = 0.0;
            fp.err = 0.0;
            fp.step = 0;
            fp.failed = 0;
            fp.stepsize = 0.0;

            fp.x_defect = Eigen::MatrixXd::Zero(dim_x, N + 1);
        }

#pragma omp section
        {
            bp.ku = Eigen::MatrixXd::Zero(dim_u, N);
            bp.Ku.resize(N, Eigen::MatrixXd::Zero(dim_u, dim_x));
            bp.ky = Eigen::MatrixXd::Zero(dim_c, N);
            bp.Ky.resize(N, Eigen::MatrixXd::Zero(dim_c, dim_x));
            bp.ks = Eigen::MatrixXd::Zero(dim_c, N);
            bp.Ks.resize(N, Eigen::MatrixXd::Zero(dim_c, dim_x));
            bp.reg = 0.0;
            bp.failed = 0;
            bp.recovery = 0.0;
            bp.opterr = 0.0;
        }
    }
}
