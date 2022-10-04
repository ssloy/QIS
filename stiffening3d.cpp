#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <ultimaille/all.h>

using Clock = std::chrono::high_resolution_clock;
using namespace std::literals::chrono_literals;
using namespace UM;

int main(int argc, char** argv) {
    constexpr double theta = 1./2.; // the energy is (1-theta)*(shape energy) + theta*(area energy)
    constexpr int bfgs_maxiter  = 30000; // max number of inner iterations
    constexpr int outer_maxiter = 3000;  // max number of outer iterations
    constexpr double bfgs_threshold  = 1e-15;
    constexpr double outer_threshold = 1e-4;

    if (3>argc) {
        std::cerr << "Usage: " << argv[0] << " init.mesh reference.mesh [result.mesh]" << std::endl;
        return 1;
    }

    std::string res_filename = "result.mesh";
    if (4<=argc) {
        res_filename = std::string(argv[3]);
    }

    Tetrahedra ref, ini;
    read_by_extension(argv[1], ref);
    read_by_extension(argv[2], ini);
//    std::cerr << "Stiffening " << argv[1] << "," << ini.nverts() << "," << std::endl;

    if (ini.nverts()!=ref.nverts() || ini.ncells()!=ref.ncells()) {
        std::cerr << "Error: " << argv[1] << " and " << argv[2] << " must have the same number of vertices and tetrahedra, aborting" << std::endl;
        return -1;
    }

    bool inverted = false;
    { // ascertain the mesh requirements
        double ref_volume = 0, ini_volume = 0;
        for (int c : cell_iter(ref)) {
            double v1 = ref.util.cell_volume(c);
            double v2 = ini.util.cell_volume(c);
            ref_volume += v1;
            ini_volume += v2;
            um_assert((v1<0 && v2<0) || (v1>0 && v2>0));
        }
        inverted = (ref_volume<=0);
        if (inverted) {
            std::cerr << "Warning: the input has negative volume, inverting" << std::endl;
            for (vec3 &p : ini.points)
                p.x *= -1;
            for (vec3 &p : ref.points)
                p.x *= -1;
        }
    }

    CellAttribute<mat<4,3>> reference(ini);    // desired tet geometry
    CellAttribute<double>   volume(ini);       // 3D triangle area
    std::vector<double> X(ini.nverts()*3, 0.); // optimization variables
    PointAttribute<bool> lock(ini, false);     // locked vertices

    for (int v : vert_iter(ini))
        for (int d : {0,1,2})
            X[3*v+d] = ini.points[v][d];

    { // lock boundary vertices
        OppositeFacet conn(ini);
        for (int c : cell_iter(ini))
            for (int lf : {0,1,2,3})
                if (conn[ini.facet(c, lf)]<0)
                    for (int lv : {0,1,2})
                        lock[ini.facet_vert(c, lf, lv)] = true;
        for (int c : cell_iter(ini)) {
            int nlck = 0;
            for (int lv : {0,1,2,3})
                nlck += lock[ini.vert(c, lv)];
//          um_assert(nlck<4);
        }
    }

    for (int t : cell_iter(ref)) { // set up reference geometry
        volume[t] = ref.util.cell_volume(t);
        mat<3,3> ST = {{
            ref.points[ref.vert(t, 1)] - ref.points[ref.vert(t, 0)],
            ref.points[ref.vert(t, 2)] - ref.points[ref.vert(t, 0)],
            ref.points[ref.vert(t, 3)] - ref.points[ref.vert(t, 0)]
        }};
        reference[t] = mat<4,3>{{ {-1,-1,-1},{1,0,0},{0,1,0},{0,0,1} }}*ST.invert_transpose();
    }

    const auto getJ = [&reference, &ini](const std::vector<double>& X, int t)->mat<3, 3> { // get Jacobian matrix for tetrahedron t
        mat<3, 3> J = {};
        for (int i : {0, 1, 2, 3})
            for (int d : {0, 1, 2})
                J[d] += reference[t][i] * X[ini.vert(t, i)*3 + d];
        return J;
    };

    const auto qual_max = [&ini, &theta, &getJ](const std::vector<double>& X)->double { // evaluate the maximum distortion over the mesh
        double t = 0;
#pragma omp parallel for reduction(max:t)
        for (int i=0; i<ini.ncells(); i++) {
            const mat<3, 3> J = getJ(X, i);
            double det = J.det();
            double f = J.sumsqr()/(3.*pow(det, 2./3.));
            double g = (1+det*det)/(2.*det);
            t = std::max(t, (det<=0 ? 1e32 : (1.-theta)*f + theta*g));
        }
        return t;
    };

    double param = (1.-1e-1)/qual_max(X);
    std::cerr << "Stiffening " << argv[1] << " - " << argv[2] << ", nverts: " << ini.nverts() << ", " << " max distortion: " << qual_max(X) << std::endl;

    auto starting_time = Clock::now();
    std::vector<SpinLock> spin_locks(X.size());
    for (int iter=0; iter<outer_maxiter; iter++) {
        std::cerr << "Outer iteration #" << iter << std::endl;
        const STLBFGS::func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
            std::fill(G.begin(), G.end(), 0);
            F = 0;
            int err = 0;
#pragma omp parallel for reduction(+:F) reduction(+:err)
            for (int t=0; t<ini.ncells(); t++) {
                if (err) continue;
                const mat<3, 3> J = getJ(X, t);
                const mat<3, 3> K = { // dual basis
                    {{
                         J[1].y*J[2].z - J[1].z*J[2].y,
                         J[1].z*J[2].x - J[1].x*J[2].z,
                         J[1].x*J[2].y - J[1].y*J[2].x
                     },
                    {
                        J[0].z*J[2].y - J[0].y*J[2].z,
                        J[0].x*J[2].z - J[0].z*J[2].x,
                        J[0].y*J[2].x - J[0].x*J[2].y
                    },
                    {
                        J[0].y*J[1].z - J[0].z*J[1].y,
                        J[0].z*J[1].x - J[0].x*J[1].z,
                        J[0].x*J[1].y - J[0].y*J[1].x
                    }}
                };
                const double det = J[0]*K[0];
                err += (det<=0);
                if (err) continue;

                double f = J.sumsqr()/(3.*pow(det, 2./3.));
                double g = (1+det*det)/(2.*det);
                const double F_ = ((1-theta)*f + theta*g);
                err += (1.<param*F_);
                if (err) continue;
                F += F_ / (1.-param*F_) * volume[t];

                for (int d : {0, 1, 2}) {
                    const vec3& a = J[d];
                    const vec3& b = K[d];
                    const vec3 dfda = (a*2.)/(3.*pow(det, 2./3.)) - b*((2.*f)/(3.*det));
                    const vec3 dgda = b*((det-g)/det);

                    for (int i : {0, 1, 2, 3}) {
                        const int v = ini.vert(t,i);
                        if (lock[v]) continue;
                        spin_locks[v*3+d].lock();
                        G[v*3+d] += (dfda*(1.-theta) + dgda*theta)/pow(1.-param*F_, 2.) * volume[t] * reference[t][i];
                        spin_locks[v*3+d].unlock();
                    }
                }
            }
            if (err) F = 1e32;
        };

        double E_prev, E;
        std::vector<double> trash(X.size());
        func(X, E_prev, trash);
        double qual_max_prev = qual_max(X);

        STLBFGS::Optimizer opt{func};
        opt.ftol = opt.gtol = bfgs_threshold;
        opt.maxiter = bfgs_maxiter;
        opt.run(X);

        func(X, E, trash);
        std::cerr << "E: " << E_prev << " --> " << E << ", t: " << param << ", max distortion: " << qual_max(X) << std::endl;

        const double sigma = std::max(1.-E/E_prev, 1e-1);
        const double qmax = qual_max(X);
        if (std::abs(qual_max_prev - qmax)/qmax<outer_threshold) break;
        param = param + sigma*(1.-param*qmax)/qmax;
    }

    std::cerr << "Running time: " << (Clock::now()-starting_time)/1.s << " seconds" << std::endl;

    for (int v : vert_iter(ini))
        ini.points[v] = {X[3*v+0], X[3*v+1], X[3*v+2]};

    if (inverted)
        for (vec3 &p : ini.points)
            p.x *= -1;
    write_by_extension(res_filename, ini, VolumeAttributes{ { {"selection", lock.ptr} }, { /*{"det", opt.det.ptr}*/ }, {}, {} });

    return 0;
}

