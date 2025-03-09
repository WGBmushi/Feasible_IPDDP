# Feasible_IPDDP
Implementation of Interior Point Differential Dynamic Programming using SymEngine.



## Dependencies
- SymEngine: Fast symbolic manipulation library to achieve automatic differentiation.
- OpenMP: Parallelization is implemented to accelerate computation. (Optional) If not enabled, please comment out the relevant interfaces.

## Build and Run
```bash
mkdir build && cd build
cmake ..
make -j4
./examples/ocp_inverted_pendulum 
```

## Results
![phase_trajectory](results/phase_trajectory.png)
![control_input](results/control_input.png)
![cost](results/cost.png)


