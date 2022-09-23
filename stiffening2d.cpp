#include <iostream>
#include <limits>
#undef NDEBUG
#include <cassert>
#include <cstring>
#include <chrono>

#define UNTANGLE 0 // use at your own risk, it is not the best implementation

#define USE_EIGEN 0
#if USE_EIGEN
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#else
#include <OpenNL_psm/OpenNL_psm.h>
#endif


#include <ultimaille/all.h>

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

struct Untangle2D {
    Untangle2D(Triangles &mesh) : m(mesh), X(m.nverts()*2), lock(m.points), ref_tri(m), J(m), K(m), det(m), area(m) {
        for (int t : facet_iter(m)) {
            area[t] = m.util.unsigned_area(t);
            vec2 A,B,C;
            m.util.project(t, A, B, C);
            um_assert(triangle_area_2d(A,B,C)>0);
            mat<2,2> ST = {{B-A, C-A}};
            ref_tri[t] = mat<3,2>{{ {-1,-1},{1,0},{0,1} }}*ST.invert_transpose();
        }
    }

    void lock_boundary_verts() {
        SurfaceConnectivity fec(m);
        for (int v : vert_iter(m))
            lock[v] = fec.is_boundary_vert(v);
    }

    void compute_hessian_pattern() {
        hessian_pattern = std::vector<int>(m.nverts());

        // enumerate all non-zero entries of the hessian matrix
        std::vector<std::tuple<int, int> > nonzero;
        for (int t : facet_iter(m))
            for (int i : range(3)) {
                int vi = m.vert(t,i);
                if (lock[vi]) continue;
                for (int j : range(3)) {
                    int vj = m.vert(t,j);
                    if (lock[vj]/* || vi<vj*/) continue;
                    nonzero.emplace_back(vi, vj);
                }
            }
        for (int i=0; i<m.nverts(); i++)
            if (lock[i])
                nonzero.emplace_back(i, i);

        // well those are not triplets, because we have stored indices only, but you get the idea
        // sort the nonzero array, and then determine the number of nonzero entries per row (the pattern)
        int ntriplets = nonzero.size();
        std::sort(nonzero.begin(), nonzero.end());
        int a=0, b=0;
        int nnz = 0;
        for (int v : vert_iter(m)) {
            a = b;
            while (b<ntriplets && std::get<0>(nonzero[++b])<v+1);
            int cnt = 1;
            for (int i=a; i<b-1; i++)
                cnt += (std::get<1>(nonzero[i]) != std::get<1>(nonzero[i+1]));
            hessian_pattern[v] = cnt;
            nnz += cnt;
        }

        if (debug>0) {
            std::cerr << "hessian matrix #non-zero entries: " << nnz << std::endl;
            std::cerr << "hessian matrix avg #nnz per row: " << double(nnz)/double(m.nverts()) << std::endl;
        }
    }

    void evaluate_jacobian(const std::vector<double> &X) {
        detmin = std::numeric_limits<double>::max();
        ninverted = 0;
#pragma omp parallel for reduction(min:detmin) reduction(+:ninverted)
        for (int t=0; t<m.nfacets(); t++) {
            mat<2,2> &J = this->J[t];
            J = {};
            for (int i=0; i<3; i++)
                for (int d : range(2))
                    J[d] += ref_tri[t][i]*X[2*m.vert(t,i) + d];
            this->K[t] = { {{ +J[1].y, -J[1].x }, { -J[0].y, +J[0].x }} };  // dual basis
            det[t] = J.det();
            detmin = std::min(detmin, det[t]);
            ninverted += (det[t]<=0);
        }
    }

    double evaluate_energy(const double eps, const std::vector<double> &X) {
        evaluate_jacobian(X);
#if !UNTANGLE
        if (ninverted>0) return 1e32;
#endif
        double E = 0;
//#pragma omp parallel for reduction(+:E)
        for (int t=0; t<m.nfacets(); t++) {
#if UNTANGLE
            double c1 = chi(eps, det[t]);
#else
            double c1 = det[t];
#endif
            double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1])/(2.*c1);
            double g = (1+det[t]*det[t])/(2.*c1);
#if UNTANGLE
            E += ((1.-theta)*f + theta*g) * area[t];
#else
            if (1.-eps*((1.-theta)*f + theta*g) < 0) return 1e32;
            E += ((1.-theta)*f + theta*g)/(1.-eps*((1.-theta)*f + theta*g)) * area[t];
#endif
        }
        return E;
    }

    void newton(const double eps, const int dim, std::vector<double> &sln) {
        int nvar = m.nverts();
        sln = std::vector<double>(nvar, 0);

#if USE_EIGEN
        std::vector<Eigen::Triplet<double> > H_triplets;
        Eigen::VectorXd G = Eigen::VectorXd::Zero(nvar);
#else
        nlNewContext();
        nlEnable(NL_NO_VARIABLES_INDIRECTION);
        nlSolverParameteri(NL_NB_VARIABLES, nvar);
        nlSolverParameteri(NL_SOLVER, NL_CG);
        nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI);
//      nlSolverParameteri(NL_SYMMETRIC, NL_TRUE);
        nlSolverParameteri(NL_MAX_ITERATIONS, NLint(nlmaxiter));
        nlSolverParameterd(NL_THRESHOLD, nlthreshold);
        nlEnable(NL_VARIABLES_BUFFER);
        nlBegin(NL_SYSTEM);

        nlBindBuffer(NL_VARIABLES_BUFFER, 0, sln.data(), NLuint(sizeof(double)));

        nlBegin(NL_MATRIX_PATTERN);
        for (auto [row, size] : enumerate(hessian_pattern))
            nlSetRowLength(row, size);
        nlEnd(NL_MATRIX_PATTERN);

        nlBegin(NL_MATRIX);
#endif

        if (debug>3) std::cerr << "preparing the matrix...";
        for (int t : facet_iter(m)) {
#if UNTANGLE
            double c1 = chi(eps, det[t]);
            double c2 = chi_deriv(eps, det[t]);
#else
            double c1 = det[t];
            double c2 = 1.;
#endif

            double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1])/(2.*c1);
            double g = (1+det[t]*det[t])/(2.*c1);

            vec2 a = J[t][dim]; // tangent basis
            vec2 b = K[t][dim]; // dual basis
            vec2 dfda = (a - b*f*c2)/c1;
            vec2 dgda = b*(det[t]-g*c2)/c1;

            for (int i=0; i<3; i++) {
                int v = m.vert(t,i);
                if (!lock[v]) {
#if UNTANGLE
                    double val = (dfda*(1.-theta) + dgda*theta) * area[t] * ref_tri[t][i];
#else
                    double val = (dfda*(1.-theta) + dgda*theta)/pow(1.-eps*((1.-theta)*f + theta*g), 2.) * area[t] * ref_tri[t][i];
#endif

#if USE_EIGEN
                    G(v) += val;
#else
                    nlAddIRightHandSide(v, val);
#endif
                }
            }


            mat<2,2> Fii = (mat<2,2>::identity() - (mat<2,1>{{{b.x},{b.y}}}*mat<1,2>{{{dfda.x,dfda.y}}} + mat<2,1>{{{dfda.x},{dfda.y}}}*mat<1,2>{{{b.x,b.y}}} )*c2   )/c1;
            mat<2,2> Gii = (mat<2,1>{{{b.x},{b.y}}}*mat<1,2>{{{b.x,b.y}}} - (mat<2,1>{{{b.x},{b.y}}}*mat<1,2>{{{dgda.x,dgda.y}}} + mat<2,1>{{{dgda.x},{dgda.y}}}*mat<1,2>{{{b.x,b.y}}})*c2 ) /c1;
#if UNTANGLE
            mat<2,2> Pii = (Fii*(1.-theta) + Gii*theta) * area[t];
#else
            mat<2,1> tmp = {{{dfda.x*(1.-theta) + dgda.x*theta},{dfda.y*(1.-theta) + dgda.y*theta }}};
            mat<2,2> Pii = ((Fii*(1.-theta) + Gii*theta)/pow(1.-eps*((1.-theta)*f + theta*g), 2.) + (tmp*tmp.transpose())*2.*eps/pow(1.-eps*((1.-theta)*f + theta*g), 3.)) * area[t];
#endif

            //TODO check incomplete cholesky by kaporin, remove locked vertices
            for (int i=0; i<3; i++) {
                int vi = m.vert(t,i);
                if (lock[vi]) continue;

                for (int j=0; j<3; j++) {
                    int vj = m.vert(t,j);
                    if (lock[vj]/* || vi<vj*/) continue;
                    double val = ref_tri[t][i]*(Pii*ref_tri[t][j]);
#if USE_EIGEN
                    H_triplets.emplace_back(vi, vj, val);
#else
                    nlAddIJCoefficient(vi, vj, val);
#endif
                }
            }
        }
        for (int v : vert_iter(m))
            if (lock[v]) {
#if USE_EIGEN
                H_triplets.emplace_back(v, v, 1);
#else
                nlAddIJCoefficient(v, v, 1);
#endif
            }
        if (debug>3) std::cerr << "ok" << std::endl;
#if USE_EIGEN
        Eigen::SparseMatrix<double, Eigen::RowMajor> H(nvar, nvar);
        H.setFromTriplets(H_triplets.begin(), H_triplets.end());
//        if (debug>1) std::cerr << "solving" << std::endl;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double> > iccg;
        iccg.compute(H);
        Eigen::VectorXd Pm = iccg.solve(G);
        if (debug>1) std::cerr << "#iterations:     " << iccg.iterations() << " estimated error: " << iccg.error()      << std::endl;
        for (int i : range(nvar)) sln[i] = Pm(i);
#else
        nlEnd(NL_MATRIX);
        nlEnd(NL_SYSTEM);
        if (debug>1) std::cerr << "solving the linear system...";
        nlSolve();
        if (debug>1) std::cerr << "ok" << std::endl;

        if (debug>1) {
            int used_iters=0;
            double elapsed_time=0.0;
            double gflops=0.0;
            double error=0.0;
            int nnz = 0;
            nlGetIntegerv(NL_USED_ITERATIONS, &used_iters);
            nlGetDoublev(NL_ELAPSED_TIME, &elapsed_time);
            nlGetDoublev(NL_GFLOPS, &gflops);
            nlGetDoublev(NL_ERROR, &error);
            nlGetIntegerv(NL_NNZ, &nnz);
            std::cerr << ("Linear solve") << "   " << used_iters << " iters in " << elapsed_time << " seconds " << gflops << " GFlop/s" << "  ||Ax-b||/||b||=" << error << std::endl;
        }
        nlDeleteContext(nlGetCurrent());
#endif
    }

    double line_search(const double eps, std::vector<double> &deltaX, std::vector<double> &deltaY) {
        if (debug>2) std::cerr << "line search...";
        double tau = 4.;
        double E = evaluate_energy(eps, X);
        std::vector<double> pts = X;
        while (tau>1e-10) {
            for (int v : vert_iter(m)) {
                X[v*2+0] = pts[v*2+0] - deltaX[v]*tau;
                X[v*2+1] = pts[v*2+1] - deltaY[v]*tau;
            }
            double E2 = evaluate_energy(eps, X);
            if (E2<E) break;
            tau /= 2.;
        }
        if (debug>2) std::cerr << "ok, tau: " << tau << std::endl;;
        return tau;
    }

    bool go() {
        compute_hessian_pattern();
#if UNTANGLE
        double qual_max = -std::numeric_limits<double>::max();
#pragma omp parallel for reduction(max:qual_max)
        for (int t=0; t<m.nfacets(); t++) {
            double c1 = det[t];
            double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1])/(2.*c1);
            double g = (1+det[t]*det[t])/(2.*c1);
            qual_max = std::max(qual_max, ((1.-theta)*f + theta*g));
        }
        double param = (1.-1e-10)/qual_max;
#else
        double param = 0;
#endif
        evaluate_jacobian(X);

        if (debug>0) std::cerr <<  "number of inverted elements: " << ninverted << std::endl;

        double qual_max_prev = std::numeric_limits<double>::max();
        for (int itero=0; itero<maxiter; itero++) {
            double E_prev = evaluate_energy(param, X);
            for (int iter=0; iter<5; iter++) {
#if !UNTANGLE
                um_assert(!ninverted);
#endif
                if (debug>1) std::cerr << "iteration #" << iter << std::endl;

                double E_prev = evaluate_energy(param, X);

                std::vector<double> deltaX, deltaY;
                newton(param, 0, deltaX);
                newton(param, 1, deltaY);
                line_search(param, deltaX, deltaY);

                double E = evaluate_energy(param, X);
                if (debug>1) std::cerr << "E: " << E << " param: " << param << " detmin: " << detmin << " ninv: " << ninverted << std::endl;

                if  (std::abs(E_prev - E)/E<1e-5) break;
            }
            double qual_max = -std::numeric_limits<double>::max();
#pragma omp parallel for reduction(max:qual_max)
            for (int t=0; t<m.nfacets(); t++) {
                double c1 = det[t];
                double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1])/(2.*c1);
                double g = (1+det[t]*det[t])/(2.*c1);
                qual_max = std::max(qual_max, ((1.-theta)*f + theta*g));
            }
            double E = evaluate_energy(param, X);
            std::cerr << "f_+: " << qual_max << /*" E_prev: " << E_prev <<*/ " E: " << E << std::endl;
#if UNTANGLE
            if (detmin>0 && std::abs(E_prev - E)/E<1e-5) break;
            double sigma = std::max(1.-E/E_prev, 2e-1);
            double mu = (1-sigma)*chi(param, detmin);
            if (detmin<mu)
                param = std::max(1e-9, 2*std::sqrt(mu*(mu-detmin)));
            else param = 1e-9;
#else
            if (std::abs(qual_max_prev - qual_max)/qual_max<1e-5/* || std::abs(E_prev - E)/E<1e-5*/) break;
            qual_max_prev = qual_max;
            double sigma = std::max(1.-E/E_prev, 1e-1);
            param = param + sigma*(1-param*qual_max)/qual_max;
#endif
        }
        return !ninverted;
    }


    void print_quality() {
        evaluate_jacobian(X);
        double qual_max = -std::numeric_limits<double>::max();
        double smin =  std::numeric_limits<double>::max();
        double smax = -std::numeric_limits<double>::max();
        for (int t : facet_iter(m)) {
            mat2x2 J = this->J[t];

            double f = (J[0]*J[0] + J[1]*J[1])/(2.*J.det());
            double g = (1+J.det()*J.det())/(2.*J.det());
            qual_max = std::max(qual_max, ((1.-theta)*f + theta*g));

            mat2x2 G = J.transpose() * J;
            mat2x2 evec;
            vec2 eval;
            eigendecompose_symmetric(G, eval, evec);
            smax = std::max(smax, std::sqrt(eval.x));
            smin = std::min(smin, std::sqrt(eval.y));
        }
        std::cerr << "f_+ = " << qual_max <<  ", t = " << 1./qual_max << std::endl;
        std::cerr << "sqrt(smax/smin) = " << std::sqrt(smax/smin) << std::endl;
    }

    ////////////////////////////////
    //      state variables       //
    ////////////////////////////////

    // optimization input parameters
    Triangles &m;           // the mesh to optimize
    double theta = 0.5; // the energy is (1-theta)*(shape energy) + theta*(area energy)
    int maxiter = 1000;    // max number of outer iterations
    int nlmaxiter = 15000;
    double nlthreshold = 1e-8;

    int debug = 1;          // verbose level

    // optimization state variables
    std::vector<double> X;     // current geometry
    PointAttribute<bool> lock; // currently lock = boundary vertices
    FacetAttribute<mat<3,2>> ref_tri;
    FacetAttribute<mat<2,2>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    FacetAttribute<mat<2,2>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    FacetAttribute<double> det; // per-tet determinant of the Jacobian matrix
    FacetAttribute<double> area; // reference area

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra

    std::vector<int> hessian_pattern; // number of non zero entries per row of the hessian matrix
};

int main(int argc, char** argv) {
    if (argc<2) {
        std::cerr << "Usage: " << argv[0] << " 3d-surface-with-texcoords-per-vertex.obj [output.obj]" << std::endl;
        return 1;
    }

    std::cerr << "Optimizing " << argv[1] << std::endl;

    std::string res_filename = "quasi-isometric-map.obj";
    if (argc>2)
        res_filename = std::string(argv[2]);

    Triangles m;
    SurfaceAttributes attr = read_by_extension(argv[1], m);
    m.delete_isolated_vertices();

    PointAttribute<vec2> tex_coord("tex_coord", attr, m);
    Untangle2D opt(m);

    for (int v : vert_iter(m))
        for (int d : range(2))
            opt.X[2*v+d] = tex_coord[v][d];

    opt.print_quality();
    auto t1 = std::chrono::high_resolution_clock::now();
    bool success = opt.go();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = t2 - t1;

    if (success) {
        opt.print_quality();
        std::cerr << "SUCCESS; running time: " << time.count() << " s; min det J = " << opt.detmin << std::endl;
    } else {
        std::cerr << "FAIL" << std::endl;
        return -1;
    }

    for (int v : vert_iter(m))
        for (int d : range(2))
            tex_coord[v][d] = opt.X[2*v+d];
    write_by_extension(res_filename, m, SurfaceAttributes{ { {"tex_coord", tex_coord.ptr} }, {}, {} });

    return 0;
}

