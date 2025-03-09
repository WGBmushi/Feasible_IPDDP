#ifndef IPDDP_HPP
#define IPDDP_HPP

#include "ipddp_ocp.hpp"
#include "ipddp_options.hpp"
#include "ipddp_utils.hpp"

#include "ipddp_backwardpass.hpp"
#include "ipddp_forwardpass.hpp"

struct Result
{
    ForwardPass fp;
    BackwardPass bp;
    std::vector<double> costs;
    double elapsedTime;
};

ForwardPass initialroll(const Functions &funcs, ForwardPass &fp);
ForwardPass resetfilter(ForwardPass &fp, AlgorithmOptions &alg);
BackwardPass resetreg(BackwardPass &bp);

Result ipddp(const Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg);

#endif // IPDDP_HPP
