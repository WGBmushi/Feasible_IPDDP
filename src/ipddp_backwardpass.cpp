#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

#include "ipddp_base.hpp"
#include "ipddp_core.hpp"

BackwardPass backwardpass(const Functions &funcs, ForwardPass &fp, BackwardPass &bp, AlgorithmOptions &alg)
{
    const int N = fp.horizon;
    const int dim_x = fp.x.rows();
    const int dim_u = fp.u.rows();
    const int dim_c = fp.c.rows();

    double c_err = 0.0;
    double mu_err = 0.0;
    double Qu_err = 0.0;

    Eigen::MatrixXd dV = Eigen::MatrixXd::Zero(2, 1);

    if (fp.failed || bp.failed)
    {
        ++bp.reg;
    }
    else if (fp.step == 0)
    {
        --bp.reg;
    }
    else if (fp.step <= 3)
    {
        bp.reg = bp.reg;
    }
    else
    {
        ++bp.reg;
    }

    if (bp.reg < 0)
    {
        bp.reg = 0;
    }
    else if (bp.reg > 24)
    {
        bp.reg = 24;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXd Vx;
    Eigen::MatrixXd Vxx;
    Eigen::MatrixXd V_1;
    Eigen::MatrixXd Vx_1;
    Eigen::MatrixXd Vxx_1;

    Eigen::VectorXd x_terminal = fp.x.col(N); 
    Eigen::VectorXd u_terminal = fp.u.col(N - 1);

    V = replaceAndEvaluate(x_terminal, u_terminal, funcs.x, funcs.u, funcs.p);
    Vx = replaceAndEvaluate(x_terminal, u_terminal, funcs.x, funcs.u, funcs.px);
    Vxx = replaceAndEvaluate(x_terminal, u_terminal, funcs.x, funcs.u, funcs.pxx);

    auto start_bp_00 = std::chrono::high_resolution_clock::now();

    std::vector<Eigen::MatrixXd> fx_list, fu_list, Qsx_list, Qsu_list;
    std::vector<Eigen::MatrixXd> qx_list, qu_list, qxx_list, qxu_list, quu_list;
    std::vector<Eigen::MatrixXd> fx_trans_list, fu_trans_list, Qsx_trans_list, Qsu_trans_list;

    std::vector<std::vector<Eigen::MatrixXd>> fxx_list, fxu_list, fuu_list;
    std::vector<std::vector<Eigen::MatrixXd>> cxx_list, cxu_list, cuu_list;

    if (alg.ddp)
    {
        precomputeMatrices_ddp(fp, funcs, fx_list, fu_list,
                               Qsx_list, Qsu_list,
                               qx_list, qu_list, qxx_list, qxu_list, quu_list,
                               fx_trans_list, fu_trans_list,
                               Qsx_trans_list, Qsu_trans_list,
                               fxx_list, fxu_list, fuu_list,
                               cxx_list, cxu_list, cuu_list);
    }
    else
    {
        precomputeMatrices_ilqr(fp, funcs, fx_list, fu_list,
                                Qsx_list, Qsu_list,
                                qx_list, qu_list, qxx_list, qxu_list, quu_list,
                                fx_trans_list, fu_trans_list,
                                Qsx_trans_list, Qsu_trans_list);
    }

    for (int i = N - 1; i >= 0; --i)
    {
        Eigen::VectorXd x = fp.x.col(i);
        Eigen::VectorXd u = fp.u.col(i);
        Eigen::VectorXd s = fp.s.col(i);
        Eigen::VectorXd y = fp.y.col(i);
        Eigen::VectorXd q = fp.q.col(i);
        Eigen::VectorXd c = fp.c.col(i);

        Eigen::MatrixXd fx = fx_list[i];
        Eigen::MatrixXd fu = fu_list[i];
        Eigen::MatrixXd Qsx = Qsx_list[i];
        Eigen::MatrixXd Qsu = Qsu_list[i];
        Eigen::MatrixXd qx = qx_list[i];
        Eigen::MatrixXd qu = qu_list[i];
        Eigen::MatrixXd qxx = qxx_list[i];
        Eigen::MatrixXd qxu = qxu_list[i];
        Eigen::MatrixXd quu = quu_list[i];
        Eigen::MatrixXd fx_trans = fx_trans_list[i];
        Eigen::MatrixXd fu_trans = fu_trans_list[i];
        Eigen::MatrixXd Qxs = Qsx_trans_list[i];
        Eigen::MatrixXd Qus = Qsu_trans_list[i];

        std::vector<Eigen::MatrixXd> fxx;
        std::vector<Eigen::MatrixXd> fxu;
        std::vector<Eigen::MatrixXd> fuu;
        std::vector<Eigen::MatrixXd> cxx;
        std::vector<Eigen::MatrixXd> cxu;
        std::vector<Eigen::MatrixXd> cuu;

        if (alg.ddp)
        {
            fxx = fxx_list[i];
            fxu = fxu_list[i];
            fuu = fuu_list[i];
            cxx = cxx_list[i];
            cxu = cxu_list[i];
            cuu = cuu_list[i];
        }

        Eigen::MatrixXd Qxx;
        Eigen::MatrixXd Qxu;
        Eigen::MatrixXd Quu;

        Eigen::MatrixXd Qx = qx + Qxs * s + fx_trans * Vx;
        // std::cout << "Qx : \n " << Qx << std::endl;

        Eigen::MatrixXd Qu = qu + Qus * s + fu_trans * Vx;
        // std::cout << "Qu : \n " << Qu << std::endl;

        if (alg.ddp)
        {
            Eigen::MatrixXd Vx_dot_fxx = tensdot(Vx, fxx);
            Eigen::MatrixXd Vx_dot_fxu = tensdot(Vx, fxu);
            Eigen::MatrixXd Vx_dot_fuu = tensdot(Vx, fuu);
            Eigen::MatrixXd cxx_dot_s = tensdot(cxx, s);
            Eigen::MatrixXd cxu_dot_s = tensdot(cxu, s);
            Eigen::MatrixXd cuu_dot_s = tensdot(cuu, s);

            Qxx = qxx + fx.transpose() * Vxx * fx + Vx_dot_fxx + cxx_dot_s;
            Qxu = qxu + fx.transpose() * Vxx * fu + Vx_dot_fxu + cxu_dot_s;
            Quu = quu + fu.transpose() * Vxx * fu + Vx_dot_fuu + cuu_dot_s;
        }
        else
        {
            Qxx = qxx + fx_trans * Vxx * fx;
            Qxu = qxu + fx_trans * Vxx * fu;
            Quu = quu + fu_trans * Vxx * fu;
        }

        Eigen::DiagonalMatrix<double, Eigen::Dynamic> S_dig = s.asDiagonal();
        Eigen::MatrixXd S = S_dig.toDenseMatrix();
        Eigen::MatrixXd Quu_reg = Quu + quu * (std::pow(1.6, bp.reg) - 1);

        Eigen::MatrixXd R;
        int failed;

        Eigen::MatrixXd r;
        Eigen::MatrixXd rhat;
        Eigen::VectorXd yinv;
        Eigen::MatrixXd SYinv;
        Eigen::MatrixXd cinv;
        Eigen::MatrixXd SCinv;
        Eigen::MatrixXd CinvS;

        Eigen::MatrixXd kK;
        Eigen::MatrixXd ku;
        Eigen::MatrixXd ks;
        Eigen::MatrixXd ky;
        Eigen::MatrixXd Ku;
        Eigen::MatrixXd Ks;
        Eigen::MatrixXd Ky;

        r = S * c + Eigen::MatrixXd::Constant(S.rows(), c.cols(), alg.mu);
        cinv = 1.0 / c.array();
        SCinv = (s.array() * cinv.array()).matrix().asDiagonal();
        CinvS = SCinv.transpose();
        std::tie(R, failed) = chol(Quu_reg - Qus * SCinv * Qsu);

        if (failed)
        {
            bp.failed = 1;
            return bp;
        }

        // -------------------------------------------------
        Eigen::MatrixXd alpha_p1 = Qu - Qus * (cinv.array() * r.array()).matrix();
        Eigen::MatrixXd beta_p1 = Qxu.transpose() - Qus * CinvS * Qsx;
        Eigen::MatrixXd Qu_combined(alpha_p1.rows(), alpha_p1.cols() + beta_p1.cols());
        Qu_combined << alpha_p1, beta_p1;
        kK = -R.transpose().llt().solve(R.llt().solve(Qu_combined));
        ku = kK.col(0);
        Ku = kK.rightCols(kK.cols() - 1);
        ks = -cinv.array() * (r.array() + (S * Qsu * ku).array());
        Ks = -CinvS * (Qsx + Qsu * Ku);
        ky = Eigen::VectorXd::Zero(dim_c);
        Ky = Eigen::MatrixXd::Zero(dim_c, dim_x);

        Quu -= Qus * SCinv * Qsu;
        Qxu -= Qxs * SCinv * Qsu;
        Qxx -= Qxs * SCinv * Qsx;
        Qu -= Qus * (cinv.array() * r.array()).matrix();
        Qx -= Qxs * (cinv.array() * r.array()).matrix();

        Eigen::MatrixXd dV_p1(2, 1);
        dV_p1 << ku.transpose() * Qu, 0.5 * ku.transpose() * Quu * ku;
        dV += dV_p1;

        Vx = Qx + Ku.transpose() * Qu + Ku.transpose() * Quu * ku + Qxu * ku;
        Vxx = Qxx + Ku.transpose() * Qxu.transpose() + Qxu * Ku + Ku.transpose() * Quu * Ku;

        bp.ku.col(i) = ku;
        bp.ky.col(i) = ky;
        bp.ks.col(i) = ks;

        bp.Ku[i] = Ku;
        bp.Ky[i] = Ky;
        bp.Ks[i] = Ks;

        Qu_err = std::max(Qu_err, Qu.lpNorm<Eigen::Infinity>());
        mu_err = std::max(mu_err, r.lpNorm<Eigen::Infinity>());
    }

    bp.failed = 0;
    bp.opterr = std::max({Qu_err, c_err, mu_err});

    return {bp};
}

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
                             std::vector<Eigen::MatrixXd> &Qsu_trans_list)
{
    int N = fp.x.cols();
    fx_list.resize(N);
    fu_list.resize(N);
    Qsx_list.resize(N);
    Qsu_list.resize(N);
    qx_list.resize(N);
    qu_list.resize(N);
    qxx_list.resize(N);
    qxu_list.resize(N);
    quu_list.resize(N);
    fx_trans_list.resize(N);
    fu_trans_list.resize(N);
    Qsx_trans_list.resize(N);
    Qsu_trans_list.resize(N);

#pragma omp parallel for
    for (int i = 0; i < N - 1; ++i)
    {
        Eigen::VectorXd x = fp.x.col(i);
        Eigen::VectorXd u = fp.u.col(i);

        fx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fx);
        fu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fu);
        Qsx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cx);
        Qsu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cu);
        qx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qx);
        qu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qu);
        qxx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qxx);
        qxu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qxu);
        quu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.quu);
        fx_trans_list[i] = fx_list[i].transpose();
        fu_trans_list[i] = fu_list[i].transpose();
        Qsx_trans_list[i] = Qsx_list[i].transpose();
        Qsu_trans_list[i] = Qsu_list[i].transpose();
    }
}

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
                            std::vector<std::vector<Eigen::MatrixXd>> &cuu_list)
{
    int N = fp.x.cols();
    fx_list.resize(N);
    fu_list.resize(N);
    Qsx_list.resize(N);
    Qsu_list.resize(N);
    qx_list.resize(N);
    qu_list.resize(N);
    qxx_list.resize(N);
    qxu_list.resize(N);
    quu_list.resize(N);
    fx_trans_list.resize(N);
    fu_trans_list.resize(N);
    Qsx_trans_list.resize(N);
    Qsu_trans_list.resize(N);
    fxx_list.resize(N);
    fxu_list.resize(N);
    fuu_list.resize(N);
    cxx_list.resize(N);
    cxu_list.resize(N);
    cuu_list.resize(N);

#pragma omp parallel for
    for (int i = 0; i < N - 1; ++i)
    {
        Eigen::VectorXd x = fp.x.col(i);
        Eigen::VectorXd u = fp.u.col(i);

        fx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fx);
        fu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fu);
        Qsx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cx);
        Qsu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cu);
        qx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qx);
        qu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qu);
        qxx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qxx);
        qxu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.qxu);
        quu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.quu);
        fx_trans_list[i] = fx_list[i].transpose();
        fu_trans_list[i] = fu_list[i].transpose();
        Qsx_trans_list[i] = Qsx_list[i].transpose();
        Qsu_trans_list[i] = Qsu_list[i].transpose();
        fxx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fxx);
        fxu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fxu);
        fuu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.fuu);
        cxx_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cxx);
        cxu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cxu);
        cuu_list[i] = replaceAndEvaluate(x, u, funcs.x, funcs.u, funcs.cuu);
    }
}
