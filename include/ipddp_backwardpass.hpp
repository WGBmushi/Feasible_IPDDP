#ifndef IPDDP_BACKARDPASS_HPP
#define IPDDP_BACKARDPASS_HPP

#include "ipddp_base.hpp"
#include "ipddp_core.hpp"

BackwardPass backwardpass(const Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg);

void precomputeMatrices_ilqr(const ForwardPass &fp, const Functions &funcs,
                             std::vector<Eigen::MatrixXd> &fx_list,
                             std::vector<Eigen::MatrixXd> &fu_list,
                             std::vector<Eigen::MatrixXd> &Qsx_list,
                             std::vector<Eigen::MatrixXd> &Qsu_list,
                             std::vector<Eigen::MatrixXd> &qx_list,
                             std::vector<Eigen::MatrixXd> &qu_list,
                             std::vector<Eigen::MatrixXd> &qxx_list,
                             std::vector<Eigen::MatrixXd> &qxu_list,
                             std::vector<Eigen::MatrixXd> &quu_list,
                             std::vector<Eigen::MatrixXd> &fx_trans_list,
                             std::vector<Eigen::MatrixXd> &fu_trans_list,
                             std::vector<Eigen::MatrixXd> &Qsx_trans_list,
                             std::vector<Eigen::MatrixXd> &Qsu_trans_list);

void precomputeMatrices_ddp(const ForwardPass &fp, const Functions &funcs,
                            std::vector<Eigen::MatrixXd> &fx_list,
                            std::vector<Eigen::MatrixXd> &fu_list,
                            std::vector<Eigen::MatrixXd> &Qsx_list,
                            std::vector<Eigen::MatrixXd> &Qsu_list,
                            std::vector<Eigen::MatrixXd> &qx_list,
                            std::vector<Eigen::MatrixXd> &qu_list,
                            std::vector<Eigen::MatrixXd> &qxx_list,
                            std::vector<Eigen::MatrixXd> &qxu_list,
                            std::vector<Eigen::MatrixXd> &quu_list,
                            std::vector<Eigen::MatrixXd> &fx_trans_list,
                            std::vector<Eigen::MatrixXd> &fu_trans_list,
                            std::vector<Eigen::MatrixXd> &Qsx_trans_list,
                            std::vector<Eigen::MatrixXd> &Qsu_trans_list,
                            std::vector<std::vector<Eigen::MatrixXd>> &fxx_list,
                            std::vector<std::vector<Eigen::MatrixXd>> &fxu_list,
                            std::vector<std::vector<Eigen::MatrixXd>> &fuu_list,
                            std::vector<std::vector<Eigen::MatrixXd>> &cxx_list,
                            std::vector<std::vector<Eigen::MatrixXd>> &cxu_list,
                            std::vector<std::vector<Eigen::MatrixXd>> &cuu_list);

#endif