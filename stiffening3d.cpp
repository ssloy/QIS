#include <iostream>
#include <fstream>
#include <limits>
#include <cassert>
#include <cstring>
#include <chrono>

#define UNTANGLE 0
#include <OpenNL_psm/OpenNL_psm.h>

#include <ultimaille/all.h>

using namespace UM;

inline double chi(double eps, double det) {
    if (det>0)
        return (det + std::sqrt(eps*eps + det*det))*.5;
    return .5*eps*eps / (std::sqrt(eps*eps + det*det) - det);
}

inline double chi_deriv(double eps, double det) {
    return .5+det/(2.*std::sqrt(eps*eps + det*det));
}

struct Untangle3D {
    Untangle3D(Tetrahedra &mesh) : m(mesh), X(m.nverts()*3), lock(m.points, false), J(m), K(m), det(m), ref_tet(m), volume(m) {
        for (int t : cell_iter(m)) {
            volume[t] = m.util.cell_volume(t);
            mat<3,3> ST = {{
                m.points[m.vert(t, 1)] - m.points[m.vert(t, 0)],
                m.points[m.vert(t, 2)] - m.points[m.vert(t, 0)],
                m.points[m.vert(t, 3)] - m.points[m.vert(t, 0)]
            }};
            ref_tet[t] = mat<4,3>{{ {-1,-1,-1},{1,0,0},{0,1,0},{0,0,1} }}*ST.invert_transpose();
        }
    }

    void lock_boundary_verts() {
        VolumeConnectivity vec(m);
        for (int c : cell_iter(m))
            for (int lf : range(4))
                if (vec.adjacent[m.facet(c, lf)]<0)
                    for (int lv : range(3))
                        lock[m.facet_vert(c, lf, lv)] = true;
    }

    void compute_hessian_pattern() {
        hessian_pattern = std::vector<int>(m.nverts());

        // enumerate all non-zero entries of the hessian matrix
        std::vector<std::tuple<int, int> > nonzero;
        for (int c=0; c<m.ncells(); c++) {
            for (int i : range(4)) {
                int vi = m.vert(c,i);
                if (lock[vi]) continue;
                for (int j : range(4)) {
                    int vj = m.vert(c,j);
                    if (lock[vj]) continue;
                    nonzero.emplace_back(vi, vj);
                }
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
        for (int c=0; c<m.ncells(); c++) {
            mat<3,3> &J = this->J[c];
            J = {};
            for (int i=0; i<4; i++)
                for (int d : range(3))
                    J[d] += ref_tet[c][i]*X[3*m.vert(c,i) + d];
            det[c] = J.det();
            detmin = std::min(detmin, det[c]);
            ninverted += (det[c]<=0);

            this->K[c] = { // dual basis
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
        }
    }

    double evaluate_energy(const double eps, const std::vector<double> &X) {
        evaluate_jacobian(X);
#if !UNTANGLE
        if (ninverted>0) return 1e32;
#endif
        double E = 0;
//#pragma omp parallel for reduction(+:E)
        for (int c=0; c<m.ncells(); c++) {
#if UNTANGLE
            double c1 = chi(eps, det[c]);
#else
            double c1 = det[c];
#endif
            double c2 = pow(c1, 2./3.);
            double f = J[c].sumsqr()/(3.*c2);
            double g = (1+det[c]*det[c])/(2.*c1);

//            double f = (J[c][0]*J[c][0] + J[c][1]*J[c][1] + J[c][2]*J[c][2])/pow(c1, 2./3.);
//            double g = (1+det[c]*det[c])/c1;

#if UNTANGLE
            E += ((1.-theta)*f + theta*g) * volume[c];
#else
            if (1.-eps*((1.-theta)*f + theta*g) < 0) return 1e32;
            E += ((1.-theta)*f + theta*g)/(1.-eps*((1.-theta)*f + theta*g)) * volume[c];
#endif
        }
        return E;
    }



    void newton(const double eps, const int dim, std::vector<double> &sln) {
        int nvar = m.nverts();
        sln = std::vector<double>(nvar, 0);

        nlNewContext();
        nlEnable(NL_NO_VARIABLES_INDIRECTION);
        nlSolverParameteri(NL_NB_VARIABLES, nvar);
        nlSolverParameteri(NL_SOLVER, NL_CG);
        nlSolverParameteri(NL_PRECONDITIONER, NL_PRECOND_JACOBI);
//      nlSolverParameteri(NL_SYMMETRIC, NL_TRUE);
        nlSolverParameteri(NL_MAX_ITERATIONS, NLint(50000));
        nlSolverParameterd(NL_THRESHOLD, 1e-20);
        nlEnable(NL_VARIABLES_BUFFER);
        nlBegin(NL_SYSTEM);

        nlBindBuffer(NL_VARIABLES_BUFFER, 0, sln.data(), NLuint(sizeof(double)));

        nlBegin(NL_MATRIX_PATTERN);
        for (auto [row, size] : enumerate(hessian_pattern))
            nlSetRowLength(row, size);
        nlEnd(NL_MATRIX_PATTERN);
        nlBegin(NL_MATRIX);

        if (debug>3) std::cerr << "preparing the matrix...";
        for (int t : cell_iter(m)) {
#if UNTANGLE
            double c1 = chi(eps, det[t]);
            double c3 = chi_deriv(eps, det[t]);
#else
            double c1 = det[t];
            double c3 = 1.;
#endif
            double c2 = pow(c1, 2./3.);

          double f = J[t].sumsqr()/(3.*c2);
          double g = (1+det[t]*det[t])/(2.*c1);

//            double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1] + J[t][2]*J[t][2])/c2;
//            double g = (1+det[t]*det[t])/c1;

            vec3 a = J[t][dim]; // tangent basis
            vec3 b = K[t][dim]; // dual basis
          vec3 dfda = (a*2.)/(3.*c2) - b*((2.*f*c3)/(3.*c1));
          vec3 dgda = b*((det[t]-g*c3)/c1);


//            vec3 dfda = a*(2./c2) - b*((2.*f*c3)/(3.*c1));
//            vec3 dgda = b*((2*det[t]-g*c3)/c1);

            for (int i=0; i<4; i++) {
                int v = m.vert(t,i);
                if (!lock[v]) {
#if UNTANGLE
                    double val = (dfda*(1.-theta) + dgda*theta) * volume[t] * ref_tet[t][i];
#else
                    double val = (dfda*(1.-theta) + dgda*theta)/pow(1.-eps*((1.-theta)*f + theta*g), 2.) * volume[t] * ref_tet[t][i];
#endif
                    nlAddIRightHandSide(v, val);
                }
            }

            mat<3,1> A = {{{a.x}, {a.y}, {a.z}}};
            mat<3,1> B = {{{b.x}, {b.y}, {b.z}}};

            mat<3,3> Fii = mat<3,3>::identity()*(2./(3.*c2)) - (A*B.transpose() + B*A.transpose())*((4.*c3)/(9.*c2*c1)) + (B*B.transpose())*f*pow(c3/c1,2)*(10./9.);
            mat<3,3> Gii = B*B.transpose()*(1. - 2.*c3*(det[t]-g*c3)/c1)/c1;

//          mat<3,3> Fii = mat<3,3>::identity()*(2./c2) - (A*B.transpose() + B*A.transpose())*((4.*c3)/(3.*c2*c1)) + (B*B.transpose())*((10.*f*c3*c3)/(9.*c1*c1));
//          mat<3,3> Gii = (B*B.transpose())*( 2./c1 - (2.*c3*(2.*det[t] - g*c3))/(c1*c1) );
#if UNTANGLE
            mat<3,3> Pii = (Fii*(1.-theta) + Gii*theta) * volume[t];
#else
            mat<3,1> tmp = {{{dfda.x*(1.-theta) + dgda.x*theta},{dfda.y*(1.-theta) + dgda.y*theta },{dfda.z*(1.-theta) + dgda.z*theta}}};
            mat<3,3> Pii = ((Fii*(1.-theta) + Gii*theta)/pow(1.-eps*((1.-theta)*f + theta*g), 2.) + (tmp*tmp.transpose())*2.*eps/pow(1.-eps*((1.-theta)*f + theta*g), 3.)) * volume[t];
#endif

//TODO OPENNL symmetric, check derivatives (AGAIN, SIGH), check incomplete cholesky by kaporin, remove locked vertices
            for (int i=0; i<4; i++) {
                int vi = m.vert(t,i);
                if (lock[vi]) continue;

                for (int j=0; j<4; j++) {
                    int vj = m.vert(t,j);
                    if (lock[vj]) continue;
                    double val = ref_tet[t][i]*(Pii*ref_tet[t][j]);
                    nlAddIJCoefficient(vi, vj, val);
                }
            }
        }
        for (int v : vert_iter(m))
            if (lock[v]) {
                nlAddIJCoefficient(v, v, 1);
            }
        if (debug>3) std::cerr << "ok" << std::endl;
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
    }

    double line_search(const double eps, std::vector<double> &deltaX, std::vector<double> &deltaY, std::vector<double> &deltaZ) {
        if (debug>2) std::cerr << "line search...";
        double tau = 4.;
        double E = evaluate_energy(eps, X);
        std::vector<double> pts = X;
        while (tau>1e-10) {
            for (int v : vert_iter(m)) {
                X[v*3+0] = pts[v*3+0] - deltaX[v]*tau;
                X[v*3+1] = pts[v*3+1] - deltaY[v]*tau;
                X[v*3+2] = pts[v*3+2] - deltaZ[v]*tau;
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
        double param = .1;
#else
        double param = 0;
#endif
        evaluate_jacobian(X);
        if (debug>0) std::cerr <<  "number of inverted elements: " << ninverted << std::endl;

        double qual_max_prev = std::numeric_limits<double>::max();
        for (int itero=0; itero<maxiter; itero++) {
            double E_prev = evaluate_energy(param, X);
            for (int iter=0; iter<5; iter++) {
                if (debug>0) std::cerr << "iteration #" << iter << std::endl;

                double E_prev = evaluate_energy(param, X);

                std::vector<double> deltaX, deltaY, deltaZ;
                newton(param, 0, deltaX);
                newton(param, 1, deltaY);
                newton(param, 2, deltaZ);
                line_search(param, deltaX, deltaY, deltaZ);

                double E = evaluate_energy(param, X);
                if (debug>0) std::cerr << "E: " << E << " param: " << param << " detmin: " << detmin << " ninv: " << ninverted << std::endl;

                if  (std::abs(E_prev - E)/E<1e-5) break;
            }
//          break;
            double qual_max = -std::numeric_limits<double>::max();
#pragma omp parallel for reduction(max:qual_max)
            for (int t=0; t<m.ncells(); t++) {
                double c1 = det[t];
                double f = (J[t][0]*J[t][0] + J[t][1]*J[t][1] + J[t][2]*J[t][2])/(3.*pow(c1, 2./3.));
                double g = (1+det[t]*det[t])/(2.*c1);
                qual_max = std::max(qual_max, ((1.-theta)*f + theta*g));
            }
            std::cerr << "qual_max: " << qual_max << std::endl;
            double E = evaluate_energy(param, X);
            std::cerr << "E_prev: " << E_prev << " E: " << E << std::endl;
#if UNTANGLE
            if (detmin>0 && std::abs(E_prev - E)/E<1e-5) break;
            double sigma = std::max(1.-E/E_prev, 2e-1);
            double mu = (1-sigma)*chi(param, detmin);
            if (detmin<mu)
                param = std::max(1e-9, 2*std::sqrt(mu*(mu-detmin)));
            else param = 1e-9;
#else
            if (std::abs(qual_max_prev - qual_max)/qual_max<1e-6 /*|| std::abs(E_prev - E)/E<1e-9*/) break;
            qual_max_prev = qual_max;
            double sigma = std::max(1.-E/E_prev, 1e-1);
            param = param + sigma*(1-param*qual_max)/qual_max;
#endif
        }
        return !ninverted;
    }


    ////////////////////////////////
    // Untangle3D state variables //
    ////////////////////////////////

    // optimization input parameters
    Tetrahedra &m;          // the mesh to optimize
    double theta = 1./2.;   // the energy is (1-theta)*(shape energy) + theta*(area energy)
    int maxiter = 1000;    // max number of outer iterations
    int bfgs_maxiter = 3000; // max number of inner iterations
    double bfgs_threshold = 1e-4;

    int debug = 1;          // verbose level

    // optimization state variables

    std::vector<double> X;     // current geometry
    PointAttribute<bool> lock; // currently lock = boundary vertices
    CellAttribute<mat<3,3>> J; // per-tet Jacobian matrix = [[JX.x JX.y, JX.z], [JY.x, JY.y, JY.z], [JZ.x, JZ.y, JZ.z]]
    CellAttribute<mat<3,3>> K; // per-tet dual basis: det J = dot J[i] * K[i]
    CellAttribute<double> det; // per-tet determinant of the Jacobian matrix
    CellAttribute<mat<4,3>> ref_tet;   // reference tetrahedron: array of 4 normal vectors to compute the gradients
    CellAttribute<double> volume; // reference volume

    double detmin;    // min(jacobian) over all tetrahedra
    int ninverted; // number of inverted tetrahedra

    std::vector<int> hessian_pattern; // number of non zero entries per row of the hessian matrix
};

static void read_lock(const std::string& filename, PointAttribute<bool>& locks) {
    std::ifstream in;
    in.open(filename, std::ifstream::in);
    if (in.fail()) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }
    std::string line;
    std::getline(in, line);
    while (!in.eof()) {
        std::getline(in, line);
        if (line == "") continue;
        std::istringstream iss(line.c_str());
        int v;
        iss >> v;
        locks[v] = true;
    }
}

int main(int argc, char** argv) {
    if (3>argc) {
        std::cerr << "Usage: " << argv[0] << " init.mesh reference.mesh [result.mesh]" << std::endl;
        return 1;
    }

    std::string res_filename = "result.mesh";
    if (4<=argc) {
        res_filename = std::string(argv[3]);
    }

    Tetrahedra ini, ref;
    read_by_extension(argv[1], ini);
    read_by_extension(argv[2], ref);
    std::cerr << "Untangling " << argv[1] << "," << ini.nverts() << "," << std::endl;

    if (ini.nverts()!=ref.nverts() || ini.ncells()!=ref.ncells()) {
        std::cerr << "Error: " << argv[1] << " and " << argv[2] << " must have the same number of vertices and tetrahedra, aborting" << std::endl;
        return -1;
    }


    bool inverted = false;
    { // ascertain the mesh requirements
        double ref_volume = 0, ini_volume = 0;
        for (int c : cell_iter(ref)) {
            ref_volume += ref.util.cell_volume(c);
            ini_volume += ini.util.cell_volume(c);
        }

        if (
                (ref_volume<0 && ini_volume>0) ||
                (ref_volume>0 && ini_volume<0)
           ) {
            std::cerr << "Error: " << argv[1] << " and " << argv[2] << " must have the orientation, aborting" << std::endl;
            return -1;
        }

        inverted = (ini_volume<=0);
        if (inverted) {
            std::cerr << "Warning: the input has negative volume, inverting" << std::endl;
            for (vec3 &p : ini.points)
                p.x *= -1;
            for (vec3 &p : ref.points)
                p.x *= -1;
        }
    }

#if 0
    vec3 bbmin, bbmax; // these are used to undo the scaling we apply to the model
    const double boxsize = 10.;

    { // scale
        ref.points.util.bbox(bbmin, bbmax);
        double maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
        for (vec3 &p : ref.points)
            p = (p - (bbmax+bbmin)/2.)*boxsize/maxside + vec3(1,1,1)*boxsize/2;
        for (vec3 &p : ini.points)
            p = (p - (bbmax+bbmin)/2.)*boxsize/maxside + vec3(1,1,1)*boxsize/2;
    }
#endif

    Untangle3D opt(ref);

    for (int v : vert_iter(ref))
        for (int d : range(3))
            opt.X[3*v+d] = ini.points[v][d];

    read_lock("../lock.txt", opt.lock);

    auto t1 = std::chrono::high_resolution_clock::now();
    bool success = opt.go();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = t2 - t1;

    if (success)
        std::cerr << "SUCCESS; running time: " << time.count() << " s; min det J = " << opt.detmin << std::endl;
    else
        std::cerr << "FAIL TO UNTANGLE!" << std::endl;

    for (int v : vert_iter(ref))
        for (int d : range(3))
            ref.points[v][d] = opt.X[3*v+d];

#if 0
    { // restore scale
        double maxside = std::max(bbmax.x-bbmin.x, bbmax.y-bbmin.y);
        for (vec3 &p : ref.points)
            p = (p - vec3(1,1,1)*boxsize/2)/boxsize*maxside + (bbmax+bbmin)/2.;
    }
#endif

    if (inverted)
        for (vec3 &p : ref.points)
            p.x *= -1;

    write_by_extension(res_filename, ref, VolumeAttributes{ { {"selection", opt.lock.ptr} }, { {"det", opt.det.ptr} }, {}, {} });
    return 0;
}

