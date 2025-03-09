#ifndef IPDDP_FORWARDPASS_HPP
#define IPDDP_FORWARDPASS_HPP

#include "ipddp_base.hpp"
#include "ipddp_core.hpp"

ForwardPass forwardpass(const Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg);

#endif