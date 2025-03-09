#ifndef IPDDP_OPTION_HPP
#define IPDDP_OPTION_HPP

#include <iostream>

struct AlgorithmOptions
{
    double tol;
    int maxiter;
    int maxiter_bp;
    double mu;
    bool ws;
    bool infeas;
    bool ddp;
    bool rigorous;

    AlgorithmOptions(
        double tolerance = 1e-5,
        int max_iterations = 1000,
        int max_iterations_bp = 20,
        double mu_value = 0.0,
        bool warm_start = false,
        bool infeasibility = false,
        bool with_3OTensor = false,
        bool rigorous = false)
        : tol(tolerance),
          maxiter(max_iterations),
          maxiter_bp(max_iterations_bp),
          mu(mu_value),
          ws(warm_start),
          infeas(infeasibility),
          ddp(with_3OTensor),
          rigorous(rigorous)
    {
    }

    void print() const
    {
        std::cout << "Tolerance value : " << tol << "\n"
                  << "Maximum number of outer iterations : " << maxiter << "\n"
                  << "maximum number of inner iterations (Backward Pass) : " << maxiter_bp << "\n"
                  << "Warm start : " << (ws ? "true" : "false") << "\n"
                  << "Handling infeasible initial guess : " << (infeas ? "true" : "false") << "\n"
                  << "Consider the third-order tensor : " << (ddp ? "true" : "false") << "\n"
                  << "Using rigorous derivations : " << (rigorous ? "true" : "false")
                  << std::endl;

        std::cout << std::endl;
    }
};

#endif // IPDDP_OPTION_HPP
