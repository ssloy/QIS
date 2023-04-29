#include <iostream>
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

int main(int argc, char** argv) {
    constexpr double theta = 1./2.; // the energy is (1-theta)*(shape energy) + theta*(area energy)
    constexpr int bfgs_maxiter  = 30000; // max number of inner iterations
    constexpr int outer_maxiter = 3000;  // max number of outer iterations
    constexpr double bfgs_threshold  = 1e-8;
    constexpr double outer_threshold = 1e-5;
    constexpr bool lock_boundary = false;
    const std::string res_filename = "result.obj";

    if (2!=argc) {
        std::cerr << "Usage: " << argv[0] << " input.obj" << std::endl;
        return 1;
    }

    Triangles m;
    SurfaceAttributes attr = read_by_extension(argv[1], m);
    PointAttribute<vec2> tex_coord("tex_coord", attr, m);
    FacetAttribute<mat<3,2>> reference(m);   // desired 2D triangle geometry
    FacetAttribute<double>   area(m);        // 3D triangle area
    std::vector<double> X(m.nverts()*2, 0.); // optimization variables
    std::vector<double> T(m.nverts(), 0.); // sub-problem optimization variables
    PointAttribute<bool> lock(m, false);     // locked vertices

    for (int t : facet_iter(m))
        um_assert(triangle_area_2d(tex_coord[m.vert(t, 0)], tex_coord[m.vert(t, 1)], tex_coord[m.vert(t, 2)])>0);

    for (int v : vert_iter(m)) // set up initialization
        for (int d : {0, 1})
            X[2*v+d] = tex_coord[v][d];

    if (lock_boundary) { // lock boundary vertices
        SurfaceConnectivity fec(m);
        for (int v : vert_iter(m))
            lock[v] = fec.is_boundary_vert(v);
    }

    for (int t : facet_iter(m)) { // set up reference geometry
        area[t] = m.util.unsigned_area(t);
        vec2 A,B,C;
        m.util.project(t, A, B, C);
        mat<2,2> ST = {{B-A, C-A}};
        reference[t] = mat<3,2>{{ {-1,-1},{1,0},{0,1} }}*ST.invert_transpose();
    }

    const auto getJ = [&reference, &m](const std::vector<double>& X, int t)->mat<2, 2> { // get Jacobian matrix for triangle t
        mat<2, 2> J = {};
        for (int i : {0, 1, 2})
            for (int d : {0, 1})
                J[d] += reference[t][i] * X[m.vert(t, i)*2 + d];
        return J;
    };

    const auto qual_max = [&m, &theta, &getJ](const std::vector<double>& X)->double { // evaluate the maximum distortion over the mesh
        double t = 0;
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

    double param = (1.-1e-1)/qual_max(X);
    std::cerr /*<< std::setprecision(std::numeric_limits<double>::max_digits10)*/ << "Stiffening " << argv[1] << ", nverts: " << m.nverts() << ", " << " max distortion: " << qual_max(X) << std::endl;

    auto starting_time = Clock::now();
    std::vector<SpinLock> spin_locks(X.size());
    for (int iter=0; iter<outer_maxiter; iter++) {
        std::cerr << "Outer iteration #" << iter << std::endl;
        const auto funcX = [&](const std::vector<double>& X, double& F, std::vector<double>& G) {
            std::fill(G.begin(), G.end(), 0);
            F = 0;
            int err = 0;
#pragma omp parallel for reduction(+:F) reduction(+:err)
            for (int t=0; t<m.nfacets(); t++) {
                if (err) continue;
                const mat<2, 2> J = getJ(X, t);
                const mat<2, 2> K = { {{ +J[1].y, -J[1].x }, { -J[0].y, +J[0].x }} };
                const double det = J[0]*K[0];
                err += (det<=0);
                if (err) continue;

                const double f = J.sumsqr()/(2.*det);
                const double g = (1+det*det)/(2.*det);
                const double F_ = ((1-theta)*f + theta*g);
                err += (1.<param*F_);
                if (err) continue;
                F += F_ / (1.-param*F_) * area[t];

                for (int d : {0, 1}) {
                    const vec2& a = J[d];
                    const vec2& b = K[d];
                    const vec2 dfda = (a - b*f)/det;
                    const vec2 dgda = b*(det - g)/det;
                    for (int i : {0, 1, 2}) {
                        const int v = m.vert(t,i);
                        if (lock[v]) continue;
                        spin_locks[v*2+d].lock();
                        G[v*2+d] += (dfda*(1.-theta) + dgda*theta)/pow(1.-param*F_, 2.) * area[t] * reference[t][i];
                        spin_locks[v*2+d].unlock();
                    }
                }
            }
            if (err) F = 1e32;
        };

        double E_prev;
        std::vector<double> GX(X.size());
	funcX(X, E_prev, GX);
        double qual_max_prev = qual_max(X);

	mat2x2 A = {};
	for (int i : {0,1})
		for (int j : {0,1})
			for (int k=0; k<m.nverts(); k++)
				A[i][j] += GX[k*2+i]*GX[k*2+j];
	mat2x2 evec;
	vec2 eval;
	eigendecompose_symmetric(A, eval, evec);
	vec2 dir = evec.col(0);
	std::vector<double> Xtmp(m.nverts()*2, 0.);

	std::cerr << std::sqrt(eval.x) << std::endl;

	const auto funcT = [&](const std::vector<double>& T, double& F, std::vector<double>& GT) {
		std::fill(GT.begin(), GT.end(), 0);
		F = 0;
		for (int k=0; k<m.nverts(); k++)
			for (int d: {0,1})
				Xtmp[k*2+d] = X[k*2+d] + T[k]*dir[d];
		funcX(Xtmp, F, GX);
#pragma omp parallel for
		for (int v=0; v<m.nverts(); v++)
			GT[v] = GX[v*2+0]*dir.x + GX[v*2+1]*dir.y;
	};


        STLBFGS::Optimizer opt{funcT};
        opt.ftol = opt.gtol = bfgs_threshold;
        opt.maxiter = bfgs_maxiter;
        opt.run(T);


	for (int k=0; k<m.nverts(); k++)
		for (int d: {0,1})
			X[k*2+d] += T[k]*dir[d];
	std::fill(T.begin(), T.end(), 0);

	double E;
        funcX(X, E, GX);
        std::cerr << "E: " << E_prev << " --> " << E << ", t: " << param << ", max distortion: " << qual_max(X) << std::endl;

        const double sigma = std::max(1.-E/E_prev, 1e-1);
        const double qmax = qual_max(X);
	if (E>1e5) break;
     //   if (std::abs(qual_max_prev - qmax)/qmax<outer_threshold) break;
        param = param + sigma*(1.-param*qmax)/qmax;
    }

    std::cerr << "Running time: " << (Clock::now()-starting_time)/1.s << " seconds" << std::endl;
    for (int v : vert_iter(m))
        tex_coord[v] = {X[2*v+0], X[2*v+1]};
    write_by_extension(res_filename, m, SurfaceAttributes{{{"tex_coord", tex_coord.ptr}, {"selection", lock.ptr}}, {}, {}});
    return 0;
}


/*
x = x0 + at

x-x0 = at
t = (x-x0)/a


f(t) = f (x0 + at)


d f(t)/dt = df(x)/dx * a
*/
