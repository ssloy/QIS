#include <iostream>
#include <limits>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <ultimaille/all.h>

using Clock = std::chrono::high_resolution_clock;
using namespace std::literals::chrono_literals;
using namespace UM;

int main(int argc, char** argv) {
    constexpr double theta = 1./2.; // the energy is (1-theta)*(shape energy) + theta*(area energy)
    constexpr int bfgs_maxiter  = 300000; // max number of inner iterations
    constexpr double bfgs_threshold  = 1e-8;
    constexpr bool lock_boundary = false;

    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " model.mesh [result.mesh]" << std::endl;
        return 1;
    }

    std::string res_filename = "result.geogram";
    if (3<=argc) {
        res_filename = std::string(argv[2]);
    }

    std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);

    Triangles m;
    SurfaceAttributes attr = read_by_extension(argv[1], m);
    PointAttribute<vec2> tex_coord("tex_coord", attr, m);

    std::vector<double> X(m.nverts()*2, 0.); // optimization variables
    PointAttribute<bool> lock(m, false);     // locked vertices
    FacetAttribute<mat<3,2>> tri_reference(m); // desired 2D triangle geometry
    FacetAttribute<double>   tri_area(m);      // 3D triangle area

    for (int v : vert_iter(m)) // set up initialization
        for (int d : {0, 1})
            X[2*v+d] = tex_coord[v][d];

    if (lock_boundary) { // lock boundary vertices
        SurfaceConnectivity fec(m);
        for (int v : vert_iter(m))
            lock[v] = fec.is_boundary_vert(v);
    }

    for (int t : facet_iter(m)) { // set up reference geometry
        tri_area[t] = m.util.unsigned_area(t);
        um_assert(tri_area[t]>0);
        vec2 A,B,C;
        m.util.project(t, A, B, C);
        mat<2,2> ST = {{B-A, C-A}};
        tri_reference[t] = mat<3,2>{{ {-1,-1},{1,0},{0,1} }}*ST.invert_transpose();
    }

    double detmin = std::numeric_limits<double>::max();
    const auto getJ = [&tri_reference, &m](const std::vector<double>& X, int t)->mat<2, 2> { // get Jacobian matrix for triangle t
        mat<2, 2> J = {};
        for (int i : {0, 1, 2})
            for (int d : {0, 1})
                J[d] += tri_reference[t][i] * X[m.vert(t, i)*2 + d];
        return J;
    };

    const auto qual_max = [&m, &theta, &getJ](const std::vector<double>& X)->double {
        double t = -std::numeric_limits<double>::max();
#pragma omp parallel for reduction(max:t)
        for (int i=0; i<m.nfacets(); i++) {
            const mat<2, 2> J = getJ(X, i);
            double det = J.det();
            double f = J.sumsqr()/(2.*det);
            double g = (1+det*det)/(2.*det);
            t = std::max(t, (det<=0 ? 1e32 : (1.-theta)*f + theta*g));
        }
        return t;
    };

    auto starting_time = Clock::now();

    std::vector<SpinLock> spin_locks(X.size());
    const STLBFGS::func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
        std::fill(G.begin(), G.end(), 0);
        F = 0;
        detmin = std::numeric_limits<double>::max();
#pragma omp parallel for reduction(+:F) reduction(min:detmin)
        for (int t=0; t<m.nfacets(); t++) {
            const mat<2, 2> J = getJ(X, t);
            const mat<2, 2> K = { {{ +J[1].y, -J[1].x }, { -J[0].y, +J[0].x }} };
            const double det = J[0]*K[0];

            detmin = std::min(detmin, det);
            if (det<=0) { F+=1e32; continue; }

            const double c1 = det;
            const double c2 = 1.;

            const double f = J.sumsqr()/(2.*c1);
            const double g = (1+det*det)/(2.*c1);
            const double F_ = ((1-theta)*f + theta*g);
//            const double s = 300.;
//            F += std::exp(s*F_) * tri_area[t];
            F += F_ * tri_area[t];
            for (int d : {0, 1}) {
                const vec2& a = J[d];
                const vec2& b = K[d];
                const vec2 dfda = (a - b*f*c2)/c1;
                const vec2 dgda = b*(det - g*c2)/c1;
                for (int i : {0, 1, 2}) {
                    const int v = m.vert(t,i);
                    if (lock[v]) continue;
                    spin_locks[v*2+d].lock();
                    double g = ((dfda*(1.-theta) + dgda*theta) * tri_reference[t][i]);
//                    G[v*2+d] += s*std::exp(s*F_)*g * tri_area[t];
                    G[v*2+d] += g * tri_area[t];
                    spin_locks[v*2+d].unlock();
                }
            }
        }
    };

    double E_prev, E;
    std::vector<double> trash(X.size());
    func(X, E_prev, trash);
    double qmaxprev = qual_max(X);

    STLBFGS::Optimizer opt{func};
    opt.ftol = bfgs_threshold;
    opt.gtol = bfgs_threshold;
    opt.maxiter = bfgs_maxiter;
    opt.run(X);

    func(X, E, trash);
    std::cerr << "Elliptic smoothing " << argv[1] << std::endl;
    std::cerr << "E: " << E_prev << " --> " << E << " qual_max: " << qmaxprev << " --> " << qual_max(X) << std::endl;
    std::cerr << "running time: " << (Clock::now()-starting_time)/1.s << " s; min det J = " << detmin << std::endl;

#if 0
    for (int v : vert_iter(m))
        m.pointsv[v] = {X[2*v+0], X[2*v+1], 0}
    write_by_extension(res_filename, m, SurfaceAttributes{ { {"selection", lock.ptr} }, {}, {} });
#else
    for (int v : vert_iter(m))
        tex_coord[v] = {X[2*v+0], X[2*v+1]};
    write_by_extension(res_filename, m, SurfaceAttributes{{{"tex_coord", tex_coord.ptr}, {"selection", lock.ptr}}, {}, {}});
#endif

    return 0;
}

