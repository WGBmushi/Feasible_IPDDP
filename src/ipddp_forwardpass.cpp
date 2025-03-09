#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

#include "ipddp_base.hpp"
#include "ipddp_core.hpp"

ForwardPass forwardpass(const Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg)
{
    int N = fp.horizon;
    int dim_x = fp.x.rows();
    int dim_u = fp.u.rows();
    int dim_c = fp.c.rows();

    int failed;
    double cost;
    double logcost;
    double err;
    double stepsize;
    double stepsize_CS;
    double stepsize_RS;
    int step;
    int step_CS;
    int step_RS;

    Eigen::MatrixXd xnew = Eigen::MatrixXd::Zero(dim_x, N + 1);
    Eigen::MatrixXd unew = Eigen::MatrixXd::Zero(dim_u, N);
    Eigen::MatrixXd ynew = Eigen::MatrixXd::Zero(dim_c, N);
    Eigen::MatrixXd snew = Eigen::MatrixXd::Zero(dim_c, N);
    Eigen::MatrixXd cnew = Eigen::MatrixXd::Zero(dim_c, N);
    Eigen::MatrixXd qnew = Eigen::MatrixXd::Zero(1, N);

    Eigen::MatrixXd xold = fp.x;
    Eigen::MatrixXd uold = fp.u;
    Eigen::MatrixXd yold = fp.y;
    Eigen::MatrixXd sold = fp.s;
    Eigen::MatrixXd cold = fp.c;

    double tau = std::max(0.99, 1 - alg.mu);

    int step_num = 11;
    std::vector<double> steplist(step_num);
    for (int i = 0; i < step_num; ++i)
    {
        steplist[i] = std::pow(2, -static_cast<double>(i));
    }

    for (step = 0; step < steplist.size(); ++step)
    {
        stepsize = steplist[step];

        failed = 0;
        xnew.col(0) = xold.col(0);

        for (int i = 0; i < N; ++i)
        {
            snew.col(i) = sold.col(i) + stepsize * bp.ks.col(i) + bp.Ks[i] * (xnew.col(i) - xold.col(i));
            unew.col(i) = uold.col(i) + stepsize * bp.ku.col(i) + bp.Ku[i] * (xnew.col(i) - xold.col(i));
            cnew.col(i) = replaceAndEvaluate(xnew.col(i), unew.col(i), funcs.x, funcs.u, funcs.c);

            if ((cnew.col(i).array() > (1 - tau) * cold.col(i).array()).any() ||
                (snew.col(i).array() < (1 - tau) * sold.col(i).array()).any())
            {
                failed = 1;
                break;
            }

            xnew.col(i + 1) = replaceAndEvaluate(xnew.col(i), unew.col(i), funcs.x, funcs.u, funcs.f);
        }

        if (failed)
        {
            continue;
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < N; ++i)
            {
                qnew.col(i) = replaceAndEvaluate(xnew.col(i), unew.col(i), funcs.x, funcs.u, funcs.q);
            }
            Eigen::VectorXd u_zero = Eigen::VectorXd::Zero(funcs.dim_u);
            Eigen::MatrixXd pnew = replaceAndEvaluate(xnew.col(N), u_zero, funcs.x, funcs.u, funcs.p);
            cost = qnew.sum() + pnew.value();

            Eigen::VectorXd u_N = Eigen::VectorXd::Zero(funcs.dim_u);
            Eigen::MatrixXd p_new_N = replaceAndEvaluate(xnew.col(N), u_N, funcs.x, funcs.u, funcs.p);

            logcost = qnew.sum() + p_new_N.value() - alg.mu * ((-cnew.array()).log()).sum();
            err = 0;

            Eigen::Vector2d candidate(logcost, err);

            if ((candidate.array() >= fp.filter.array()).all())
            {
                failed = 2;
                continue;
            }
            else
            {
                fp.filter = candidate;
                break;
            }
        }
    }

    if (failed)
    {
        fp.failed = failed;
        fp.stepsize = 0.0;
    }
    else
    {
        fp.cost = cost;
        fp.logcost = logcost;
        fp.x = xnew;
        fp.u = unew;
        fp.y = ynew;
        fp.s = snew;
        fp.c = cnew;
        fp.q = qnew;
        fp.err = err;
        fp.stepsize = stepsize;
        fp.step = step;
        fp.failed = 0;
    }

    return fp;
}
