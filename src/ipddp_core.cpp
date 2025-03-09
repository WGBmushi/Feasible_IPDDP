#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>

#include "ipddp_base.hpp"
#include "ipddp_core.hpp"

Result ipddp(const Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg)
{
    auto start_init = std::chrono::high_resolution_clock::now();

    std::vector<double> costs;
    std::vector<double> steps;

    fp = initialroll(funcs, fp);

    if (alg.mu == 0)
    {
        alg.mu = fp.cost / fp.horizon / fp.s.rows();
    }

    fp = resetfilter(fp, alg);

    bp = resetreg(bp);

    // ------------------------------

    int iter = 0;
    for (iter; iter < alg.maxiter; ++iter)
    {
        auto start_iter = std::chrono::high_resolution_clock::now();

        int iter_bp = 0;

        while (true)
        {
            if (iter_bp < alg.maxiter_bp)
            {
                bp = backwardpass(funcs, fp, bp, alg);
                ++iter_bp;

                if (!bp.failed)
                    break;
            }
            else
            {
                std::cerr << "Backwardpass reached maximum iterations." << std::endl;
                return {};
            }
        }

        fp = forwardpass(funcs, fp, bp, alg);
        auto end_iter = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_iter - start_iter;
        double time_iter = elapsed.count();

        if (iter % 100 == 0)
        {
            std::cout << "\n"
                      << std::setw(12) << std::left << "Iteration"
                      << std::setw(15) << std::left << "Time"
                      << std::setw(15) << std::left << "mu"
                      << std::setw(15) << std::left << "Cost"
                      << std::setw(15) << std::left << "Opt. error"
                      << std::setw(12) << std::left << "Reg. power"
                      << std::setw(12) << std::left << "Stepsize"
                      << "\n";
        }
        std::cout << std::setw(12) << std::left << iter
                  << std::setw(15) << std::left << time_iter
                  << std::setw(15) << std::left << alg.mu
                  << std::setw(15) << std::left << fp.cost
                  << std::setw(15) << std::left << bp.opterr
                  << std::setw(12) << std::left << bp.reg
                  << std::setw(12) << std::left << fp.stepsize
                  << "\n";

        costs.push_back(fp.cost);
        steps.push_back(fp.stepsize);

        if (std::max(bp.opterr, alg.mu) <= alg.tol)
        {
            std::cout << std::endl;
            std::cout << "Optimality reached." << std::endl;
            break;
        }

        if (bp.opterr <= 0.2 * alg.mu)
        {
            alg.mu = std::max(alg.tol / 10, std::min(0.2 * alg.mu, std::pow(alg.mu, 1.2)));
            fp = resetfilter(fp, alg);
            bp = resetreg(bp);
        }
    }

    if (iter >= alg.maxiter)
    {
        std::cerr << "Optimization procedure reaches the maximum iteration." << std::endl;
        return {};
    }

    auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_init).count() / 1000.0;
    std::cout << std::endl;
    std::cout << "Total elapsed time : " << elapsedTime << " sec." << std::endl;
    std::cout << std::endl;

    return {fp, bp, costs, elapsedTime};
}

ForwardPass initialroll(const Functions &funcs, ForwardPass &fp)
{
    int N = fp.horizon;
    Eigen::VectorXd x;
    Eigen::VectorXd u;

    for (int i = 0; i < N; ++i)
    {
        x = fp.x.col(i);
        u = fp.u.col(i);

        fp.c.col(i) = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.c);

        fp.q.col(i) = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.q);

        fp.x.col(i + 1) = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.f);
    }

    Eigen::VectorXd p_input1 = fp.x.col(N);
    Eigen::VectorXd p_input2 = Eigen::VectorXd::Zero(1);
    fp.cost = fp.q.sum() + replaceAndEvaluate(p_input1, p_input2, funcs.x, funcs.u, funcs.p).sum();

    return fp;
}

ForwardPass resetfilter(ForwardPass &fp, AlgorithmOptions &alg)
{
    Eigen::MatrixXd fpc_neg = -fp.c;

    Eigen::RowVectorXd fpc_row_vec = Eigen::Map<Eigen::RowVectorXd>(fpc_neg.data(), fpc_neg.size());

    Eigen::RowVectorXd fpc_log_vec = fpc_row_vec.array().log();

    double fpc_log_vec_sum = fpc_log_vec.sum();

    fp.logcost = fp.cost - alg.mu * fpc_log_vec_sum;

    fp.err = 0;

    fp.filter(0) = fp.logcost;
    fp.filter(1) = fp.err;
    fp.step = 0;
    fp.failed = 0;

    return fp;
}

BackwardPass resetreg(BackwardPass &bp)
{
    bp.reg = 0;
    bp.failed = 0;
    bp.recovery = false;
    return bp;
}
