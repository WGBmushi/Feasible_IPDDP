#include <iostream>

#include "ipddp_base.hpp"
#include "ipddp_core.hpp"

#include "ocp_inverted_pendulum.hpp"

int main()
{
  AlgorithmOptions alg;
  INVP_OCP Invp_ocp;
  Functions funcs;
  ForwardPass fp;
  BackwardPass bp;

  dynamics(Invp_ocp, funcs, fp, bp, alg);

  Result result = ipddp(funcs, fp, bp, alg);

  plot_phase_trajectory(result.fp.x);
  plot_control_input(result.fp.u);
  plot_cost(result.costs);

  return 0;
}
