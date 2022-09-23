#include <iostream>
#include <limits>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <ultimaille/all.h>

using Clock = std::chrono::high_resolution_clock;
using namespace std::literals::chrono_literals;
using namespace UM;

double triangle_area_2d(vec2 a, vec2 b, vec2 c) {
    return .5*((b.y-a.y)*(b.x+a.x) + (c.y-b.y)*(c.x+b.x) + (a.y-c.y)*(a.x+c.x));
}

double triangle_aspect_ratio_2d(vec2 a, vec2 b, vec2 c) {
    double l1 = (b-a).norm();
    double l2 = (c-b).norm();
    double l3 = (a-c).norm();
    double lmax = std::max(l1, std::max(l2, l3));
    return lmax*(l1+l2+l3)/(4.*std::sqrt(3.)*triangle_area_2d(a, b, c));
}

inline double chi(double eps, double det) {
    if (det>0)
        return (det + std::sqrt(eps*eps + det*det))*.5;
    return .5*eps*eps / (std::sqrt(eps*eps + det*det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5+det/(2.*std::sqrt(eps*eps + det*det));
}

int main(int argc, char** argv) {
    constexpr double theta = 1./2.; // the energy is (1-theta)*(shape energy) + theta*(area energy)
    constexpr int bfgs_maxiter  = 30000; // max number of inner iterations
    constexpr int outer_maxiter = 3000; // max number of outer iterations
    constexpr double bfgs_threshold  = 1e-8;
    constexpr double outer_threshold = 1e-3;
    constexpr bool lock_boundary = false;
    constexpr bool stiffen = true;

    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " model.mesh [result.mesh]" << std::endl;
        return 1;
    }

    std::string res_filename = "result.mesh";
    if (3<=argc) {
        res_filename = std::string(argv[2]);
    }

    std::cerr << std::setprecision(std::numeric_limits<double>::max_digits10);

    Triangles m;
    SurfaceAttributes attr = read_by_extension(argv[1], m);
    PointAttribute<vec2> tex_coord("tex_coord", attr, m);

#if 0
    { // scale the input geometry to have the same area as the target domain
        double target_area = 0, source_area = 0;
        for (int t : facet_iter(m)) {
            vec2 a = tex_coord[m.vert(t, 0)];
            vec2 b = tex_coord[m.vert(t, 1)];
            vec2 c = tex_coord[m.vert(t, 2)];
            double area = triangle_area_2d(a, b, c);
            if (stiffen)
                um_assert(area>0);
            target_area += area;
            source_area += m.util.unsigned_area(t);
        }

        um_assert(target_area>0); // ascertain mesh requirements
        for (vec3 &p : m.points)
            p *= std::sqrt(target_area/source_area);
    }
#endif

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

#if 0
        double ar = triangle_aspect_ratio_2d(A, B, C);
        if (ar>10) { // if the aspect ratio is bad, assign an equilateral reference triangle
            double a = ((B-A).norm() + (C-B).norm() + (A-C).norm())/3.; // edge length is the average of the original triangle
            tri_area[t] = sqrt(3.)/4.*a*a;
            A = {0., 0.};
            B = {a, 0.};
            C = {a/2., std::sqrt(3.)/2.*a};
        }
#endif

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

    int ninverted = 0;
    for (int t : facet_iter(m)) {
        double det = getJ(X, t).det();
        detmin = std::min(detmin, det);
        ninverted += (det<=0);
    }
    double param = 0;
    if (stiffen) {
        double f = qual_max(X);
        param = (1.-1e-1)/f;
//        param = 99e-2/f;
        std::cerr << "Stiffening " << argv[1] << ", nverts: " << m.nverts() << ", " << " qual_max: " << f << std::endl;
//    std::cerr << "T0 " << argv[1] << ", nverts: " << m.nverts() << ", " << " qual_max: " << qual_max(X) << std::endl;
    } else {
        param = detmin > 0. ? 1e-9 : std::min(1., std::sqrt(1e-8 + 1e-4*detmin*detmin));
        std::cerr << "Untangling " << argv[1] << ", nverts: " << m.nverts() << ", " << " ninverted: " << ninverted << std::endl;
    }

    auto starting_time = Clock::now();

    std::vector<SpinLock> spin_locks(X.size());
    for (int iter=0; iter<outer_maxiter; iter++) {
        std::cerr << "iteration #" << iter << std::endl;
        const STLBFGS::func_grad_eval func = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
            std::fill(G.begin(), G.end(), 0);
            F = 0;
            ninverted = 0;
            detmin = std::numeric_limits<double>::max();
#pragma omp parallel for reduction(+:F) reduction(min:detmin) reduction(+:ninverted)
            for (int t=0; t<m.nfacets(); t++) {
                const mat<2, 2> J = getJ(X, t);
                const mat<2, 2> K = { {{ +J[1].y, -J[1].x }, { -J[0].y, +J[0].x }} };
                const double det = J[0]*K[0];

                detmin = std::min(detmin, det);
                ninverted += (det<=0);
                if (stiffen && ninverted) continue;

                const double c1 = stiffen ? det : chi(param, det);
                const double c2 = stiffen ? 1.  : chi_deriv(param, det);

                const double f = J.sumsqr()/(2.*c1);
                const double g = (1+det*det)/(2.*c1);
                const double F_ = ((1-theta)*f + theta*g);
                if (stiffen) {
                    if (1.<param*F_)
                        ninverted++;
                    else F += F_ / (1.-param*F_) * tri_area[t];
                } else
                    F += F_ * tri_area[t];

                if (stiffen && ninverted) continue;

                for (int d : {0, 1}) {
                    const vec2& a = J[d];
                    const vec2& b = K[d];
                    const vec2 dfda = (a - b*f*c2)/c1;
                    const vec2 dgda = b*(det - g*c2)/c1;
                    for (int i : {0, 1, 2}) {
                        const int v = m.vert(t,i);
                        if (lock[v]) continue;
                        spin_locks[v*2+d].lock();
                        if (stiffen)
                            G[v*2+d] += (dfda*(1.-theta) + dgda*theta)/pow(1.-param*F_, 2.) * tri_area[t] * tri_reference[t][i];
                        else
                            G[v*2+d] += ((dfda*(1.-theta) + dgda*theta) * tri_reference[t][i]) * tri_area[t];
                        spin_locks[v*2+d].unlock();
                    }
                }
            }
            if (stiffen && ninverted) F = 1e32;
//            std::cerr << (stiffen && ninverted) << " " << F << std::endl;
        };

        double E_prev, E;
        std::vector<double> trash(X.size());
        func(X, E_prev, trash);
        double qual_max_prev = qual_max(X);

        STLBFGS::Optimizer opt{func};
//        LBFGS_Optimizer opt{func};
//      opt.invH.history_depth = 20;
        opt.ftol = bfgs_threshold;
        opt.gtol = bfgs_threshold;
        opt.maxiter = bfgs_maxiter;
        opt.run(X);

        func(X, E, trash);
        if (stiffen)
            std::cerr << "E: " << E_prev << " --> " << E << " t: " << param << " qual_max: " << qual_max(X) << std::endl;
        else
            std::cerr << "E: " << E_prev << " --> " << E << " eps: " << param << " detmin: " << detmin << " ninv: " << ninverted << std::endl;

//      break;

        const double sigma = std::max(1.-E/E_prev, 1e-1);
        if (stiffen) {
            const double qmax = qual_max(X);
            if (std::abs(qual_max_prev - qmax)/qmax<outer_threshold) break;
            param = param + sigma*(1.-param*qmax)/qmax;
        } else {
            if (detmin>0 && std::abs(E_prev - E)/E<outer_threshold) break;
            const double mu = (1.-sigma)*chi(param, detmin);
            if (detmin<mu)
                param = std::max(1e-9, 2*std::sqrt(mu*(mu-detmin)));
            else param = 1e-9;
        }
    }

    if (!ninverted)
        std::cerr << "SUCCESS; running time: " << (Clock::now()-starting_time)/1.s << " s; min det J = " << detmin << std::endl;
    else
        std::cerr << "FAIL TO UNTANGLE!" << std::endl;

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

