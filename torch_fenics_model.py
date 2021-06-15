# import torch

from dolfin import *
from fenics_adjoint import *
import torch_fenics
from ufl import nabla_div


import matplotlib.pyplot as plt
import numpy as np
# import logging


W = 0.2
L = 1


class HomogeneousBeam(torch_fenics.FEniCSModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()
        #TODO: make L, W variables
        # Scaled variables
        self.bc = None
        self.L = L
        self.W = W

        self.rho = 1
        self.delta = self.W/self.L
        self.gamma = 0.4*self.delta**2
        self.g = self.gamma
        self.tol = 1e-14 # boundary condition

        # Create function space                                        # of cells in x-y-z
        self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
        self.V = VectorFunctionSpace(self.mesh, 'P', 1)
        self.V0 = FunctionSpace(self.mesh, 'CG', 0) # density map

        # Create trial and test functions
        self.v = TestFunction(self.V)
        self.T = Constant((0, 0, 0)) 


    # Define strain and stress
    def epsilon(self, u):
        return 0.5*(nabla_grad(u) + nabla_grad(u).T)

    def sigma(self, u):
        return self.lambda_*nabla_div(u)*Identity(self.d) + 2*self.mu*self.epsilon(u)


    # BC's condition excerpted from https://fenicsproject.org/qa/13149/clamped-beam-under-the-end-load-problem/
    @staticmethod
    def clamped_boundary_left(x, on_boundary):
        tol = 1E-14 # tolerance is manually defined here as 1e-14
        return on_boundary and near(x[0], 0., tol) # should be using self.tol
    
    @staticmethod
    def clamped_boundary_right(x, on_boundary):
        tol = 1E-14# tolerance is manually defined here as 1e-14
        return on_boundary and near(x[0], 1., tol) # should be using self.tol


    def solve(self, mu, beta, force):
        # Parameters to be optimized
        self.mu = mu
        self.beta = beta
        self.force = force
        self.lambda_ = self.beta
        total_force = -self.rho * self.g - self.force
        self.f = Constant((0, 0, total_force))


        self.u = TrialFunction(self.V)
        self.d = self.u.geometric_dimension()  # space dimension
        self.a = inner(self.sigma(self.u), self.epsilon(self.v))*dx

        L = dot(self.f, self.v)*dx + dot(self.T, self.v)*ds

        # Construct boundary condition
        self.bc_l = DirichletBC(self.V, Constant((0, 0, 0)), self.clamped_boundary_left)

        # Solve
        self.u = Function(self.V) # displacement
        solve(self.a == L, self.u, self.bc_l)

        # Displacement
        return self.u

    def input_templates(self):
        # Declare input shapess
        return Constant((1)), Constant((1)), Constant((1))

    def exec(self, expression):
        # To work with fenics' functions
        if 'return' in expression: return exec(expression.split(' ')[1])
        else: exec(expression)


class plot_only(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()
    # Algorithmic parameters
        self.niternp = 20 # number of non-penalized iterations
        self.niter = 1 # total number of iterations
        self.pmax = 4        # maximum SIMP exponent
        self.exponent_update_frequency = 4 # minimum number of steps between exponent update
        self.tol_mass = 1e-4 # tolerance on mass when finding Lagrange multiplier
        self.thetamin = 0.01 # minimum density modeling void


        # Problem parameters
        self.thetamoy = 0.4 # target average material density
        self.E = Constant(1)
        self.nu = Constant(0.3)
        self.lamda = self.E*self.nu/(1+self.nu)/(1-2*self.nu)
        self.mu = self.E/(2*(1+self.nu))
        self.f = Constant((0, -1)) # vertical downwards force

        # Mesh
        self.mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 40, 30, "crossed")

        self.facets = MeshFunction("size_t", self.mesh, 1)
        AutoSubDomain(self.load).mark(self.facets, 1)
        self.ds = Measure("ds", subdomain_data=self.facets)

        # Function space for density field
        self.V0 = FunctionSpace(self.mesh, "DG", 0)
        # Function space for displacement
        self.V2 = VectorFunctionSpace(self.mesh, "CG", 2)

        self.p = Constant(1)  # SIMP penalty exponent
        self.exponent_counter = 0  # exponent update counter
        self.lagrange = Constant(1)  # Lagrange multiplier for volume constraint

        self.thetaold = Function(self.V0, name="Density")
        self.thetaold.interpolate(Constant(self.thetamoy))
        self.coeff = self.thetaold ** self.p
        self.theta = Function(self.V0)

        self.volume = assemble(Constant(1.) * dx(domain=self.mesh))
        self.avg_density_0 = assemble(self.thetaold * dx) / self.volume  # initial average density
        self.avg_density = 0.

        # Inhomogeneous elastic variational problem
        self.u1 = Function(self.V2, name="Old Displacement")
        self.old_compliance = 1e30
        self.ffile = XDMFFile("topology_optimization.xdmf")
        self.ffile.parameters["flush_output"] = True
        self.ffile.parameters["functions_share_mesh"] = True
        self.compliance_history = []



    def solve(self, peta):
        self.u_ = TestFunction(self.V2)
        self.u = Function(self.V2, name="Displacement")
        self.du = TrialFunction(self.V2)
        self.a = inner(self.sigma(self.u_), self.eps(self.du)) * dx
        L = dot(self.f, self.u_) * self.ds(1)


        # Fixed boundary condtions
        self.bc = DirichletBC(self.V2, Constant((0, 0)), self.left)

        # Assigning density field
        # print('before ', self.theta.vector().get_local())
        # self.thetaold.vector().set_local(peta.values())  # this is a single value lol!


        self.thetaold.vector()[:] = peta.values()
        # self.thetaold = peta
        # self.thetaold = peta
        # print('after ', self.theta.vector().get_local())

        # # solve
        # for i in range(self.niter):
        #     self.i = i
        #     solve(self.a == L, self.u, self.bc, solver_parameters={"linear_solver": "cg", "preconditioner": "hypre_amg"})
        #
        #     self.ffile.write(self.thetaold, i)
        #     self.ffile.write(self.u, i)
        #     # if i == 0: self.u1.assign(self.u)
        #     # if i == 1:
        #     #     difference = self.u - self.u1
        #     #     plot(-difference, 'difference in displacement field')
        #     #     plt.show();
        #     self.compliance = assemble(action(L, self.u))
        #     self.compliance_history.append(self.compliance)
        #     print("Iteration {}: compliance =".format(i), self.compliance)
        #     # print("theta before update", self.theta.vector().get_local())
        #     avg_density = self.update_theta()
        #     # print("theta after update", self.theta.vector().get_local())
        #     print('average_density', avg_density)
        #
        #     self.update_lagrange_multiplier(avg_density)
        #
        #     self.exponent_counter = self.update_exponent(self.exponent_counter)
        #
        #     # Update theta field and compliance
        #     if i % 5 == 0:
        # self.thetaold.assign(self.theta)
        plot(self.thetaold, title = 'Initialized Density Field', cmap = 'bone_r')
        plt.show();

            # self.old_compliance = self.compliance

        # return self.theta, \
        return self.thetaold


    def input_templates(self):
        # return Function(self.V2)
        return Constant([0]*4800)
        # return Constant([0]*19401*2)
    # Boundaries
    @staticmethod
    def left(x, on_boundary):
        return near(x[0], -2) and on_boundary

    @staticmethod
    def load(x, on_boundary):
        return near(x[0], 2) and near(x[1], 0.5, 0.05)

    @staticmethod
    def eps(v):
        return sym(grad(v))

    def sigma(self, v):
        return self.coeff*(self.lamda*div(v)*Identity(2)+2*self.mu*self.eps(v))

    def energy_density(self, u, v):
        return inner(self.sigma(u), self.eps(v))

    @staticmethod
    def local_project(v, V):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        u = Function(V)
        solver.solve_local_rhs(u)
        return u

    def update_theta(self):
        self.theta.assign(self.local_project((self.p*self.coeff*self.energy_density(self.u, self.u)/self.lagrange)**(1/(self.p+1)), self.V0))
        thetav = self.theta.vector().get_local()
        self.theta.vector().set_local(np.maximum(np.minimum(1, thetav), self.thetamin))
        self.theta.vector().apply("insert")
        avg_density = assemble(self.theta*dx)/self.volume
        return avg_density


    def update_lagrange_multiplier(self, avg_density):
        avg_density1 = avg_density
        # Initial bracketing of Lagrange multiplier
        if (avg_density1 < self.avg_density_0):
            lagmin = float(self.lagrange)
            while (avg_density < self.avg_density_0):
                self.lagrange.assign(Constant(self.lagrange/2))
                avg_density = self.update_theta()
            lagmax = float(self.lagrange)
        elif (avg_density1 > self.avg_density_0):
            lagmax = float(self.lagrange)
            while (avg_density > self.avg_density_0):
                self.lagrange.assign(Constant(self.lagrange*2))
                avg_density = self.update_theta()
            lagmin = float(self.lagrange)
        else:
            lagmin = float(self.lagrange)
            lagmax = float(self.lagrange)

        # Dichotomy on Lagrange multiplier
        inddico=0
        while ((abs(1.-avg_density/self.avg_density_0)) > self.tol_mass):
            self.lagrange.assign(Constant((lagmax+lagmin)/2))
            avg_density = self.update_theta()
            inddico += 1;
            if (avg_density < self.avg_density_0):
                lagmin = float(self.lagrange)
            else:
                lagmax = float(self.lagrange)
        # print("   Dichotomy iterations:", inddico)


    def update_exponent(self, exponent_counter):
        exponent_counter += 1
        if (self.i < self.niternp):
            self.p.assign(Constant(1))
        elif (self.i >= self.niternp):
            if self.i == self.niternp:
                print("\n Starting penalized iterations\n")
            if ((abs(self.compliance-self.old_compliance) < 0.01*self.compliance_history[0]) and
                (exponent_counter > self.exponent_update_frequency) ):
                # average gray level
                gray_level = assemble((self.theta-self.thetamin)*(1.-self.theta)*dx)*4/self.volume
                self.p.assign(Constant(min(float(self.p)*(1+0.3**(1.+gray_level/2)), self.pmax)))
                exponent_counter = 0
                print("   Updated SIMP exponent p = ", float(self.p))
        return exponent_counter



class varied_density_field(torch_fenics.FEniCSModule):
    def __init__(self):
        super().__init__()
    # Algorithmic parameters
        self.niternp = 20 # number of non-penalized iterations
        self.niter = 1 # total number of iterations
        self.pmax = 1        # maximum SIMP exponent
        self.exponent_update_frequency = 4 # minimum number of steps between exponent update
        self.tol_mass = 1e-4 # tolerance on mass when finding Lagrange multiplier
        self.thetamin = 0.01 # minimum density modeling void


        # Problem parameters
        self.thetamoy = 0.5 # target average material density
        self.E = Constant(20 * 1e9)
        self.nu = Constant(0.2)
        self.lamda = self.E*self.nu/(1+self.nu)/(1-2*self.nu)
        self.mu = self.E/(2*(1+self.nu))

        # Mesh
        self.mesh = RectangleMesh(Point(-2, 0), Point(2, 1), 40, 30, "crossed")

        self.facets = MeshFunction("size_t", self.mesh, 1)

        # Function space for density field
        self.V0 = FunctionSpace(self.mesh, "DG", 0)
        # Function space for displacement
        self.V2 = VectorFunctionSpace(self.mesh, "CG", 2)

        self.p = Constant(1)  # SIMP penalty exponent
        self.exponent_counter = 0  # exponent update counter
        self.lagrange = Constant(1)  # Lagrange multiplier for volume constraint

        # self.thetaold = Function(self.V0, name="Density")
        # self.thetaold.interpolate(Constant(self.thetamoy))
        # self.coeff = self.thetaold ** self.p
        # self.theta = Function(self.V0)

        # self.volume = assemble(Constant(1.) * dx(domain=self.mesh))
        # self.avg_density_0 = assemble(self.thetaold * dx) / self.volume  # initial average density
        self.avg_density = 0.

        # Inhomogeneous elastic variational problem
        self.old_compliance = 1e30
        self.ffile = XDMFFile("topology_optimization.xdmf")
        self.ffile.parameters["flush_output"] = True
        self.ffile.parameters["functions_share_mesh"] = True
        self.compliance_history = []


    def solve(self, peta):
        self.f = Constant((0, -22e3))  # vertical downwards force

        AutoSubDomain(self.load).mark(self.facets, 1)
        self.ds = Measure("ds", subdomain_data=self.facets)

        self.thetaold = peta
        # self.thetaold.interpolate(Constant(self.thetamoy))
        self.coeff = self.thetaold ** self.p
        self.theta = Function(self.V0)

        self.volume = assemble(Constant(1.) * dx(domain=self.mesh))
        self.avg_density_0 = assemble(self.thetaold * dx) / self.volume  # initial average density
        self.avg_density = 0.

        # plot(self.thetaold, 'theta', cmap='bone_r')
        # plt.show();
        # #

        self.u1 = Function(self.V2, name="Old Displacement")
        self.u_ = TestFunction(self.V2)
        self.u = Function(self.V2, name="Displacement")
        self.du = TrialFunction(self.V2)
        self.a = inner(self.sigma(self.u_), self.eps(self.du)) * dx
        L = dot(self.f, self.u_) * self.ds(1)


        # Fixed boundary condtions
        self.bc = DirichletBC(self.V2, Constant((0, 0)), self.left)

        # Assigning density field
        # self.theta.vector().set_local(peta.vector().get_local())
        # self.thetaold.vector().set_local(peta.vector().get_local())
        # self.theta = peta


        # print('after ', self.theta.vector().get_local())

        # solve
        for i in range(self.niter):
            self.i = i
            solve(self.a == L, self.u, self.bc)
            self.ffile.write(self.thetaold, i)
            self.ffile.write(self.u, i)

            self.compliance = assemble(action(L, self.u))
            self.compliance_history.append(self.compliance)
            # print("Iteration {}: compliance =".format(i), self.compliance)

            # avg_density = self.update_theta()
            # print('average_density', avg_density)

            # self.update_lagrange_multiplier(avg_density)

            # self.exponent_counter = self.update_exponent(self.exponent_counter)

            # Plotting and verifying the displacement
            # if i % 5 == 0:
            #     # self.thetaold.assign(self.theta)
            # plot(self.thetaold, 'theta', cmap = 'bone_r')
            # plt.show();
            # #
            #     plot(self.u, 'displacement')
            #     plt.show();
            #     with open('fake_displacement.txt', 'w') as f:
            #         for v in self.u.vector().get_local():
            #             f.write(str(v))
            #             f.write('\n')

            # Update theta field and compliance
            self.old_compliance = self.compliance


        # self.w0 = self.u.compute_vertex_values(self.mesh) # 5000
        # print('type of w0', type(self.w0))

        # print('len of w0', len(w0))
        # self.u.vector().set_local([float(i) for i in self.w0])
        # self.u1.vector()[:] = self.w0
        # print('type of u', type(self.u))

        # return self.theta, self.u
        return self.u


    def input_templates(self):
        return Function(self.V0)

    # Boundaries
    @staticmethod
    def left(x, on_boundary):
        return near(x[0], -2) and on_boundary

    @staticmethod
    def load(x, on_boundary):
        # return near(x[0], 2) and near(x[1], 0.5, 0.05)
        return near(x[0], 2) and near(x[1], 1 - DOLFIN_EPS, 1)

    @staticmethod
    def eps(v):
        return sym(grad(v))

    def sigma(self, v):
        return self.coeff*(self.lamda*div(v)*Identity(2)+2*self.mu*self.eps(v))

    def energy_density(self, u, v):
        return inner(self.sigma(u), self.eps(v))

    @staticmethod
    def local_project(v, V):
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        u = Function(V)
        solver.solve_local_rhs(u)
        return u

    def update_theta(self):
        self.theta.assign(self.local_project((self.p*self.coeff*self.energy_density(self.u, self.u)/self.lagrange)**(1/(self.p+1)), self.V0))
        thetav = self.theta.vector().get_local()
        self.theta.vector().set_local(np.maximum(np.minimum(1, thetav), self.thetamin))
        self.theta.vector().apply("insert")
        avg_density = assemble(self.theta*dx)/self.volume
        return avg_density


    def update_lagrange_multiplier(self, avg_density):
        avg_density1 = avg_density
        # Initial bracketing of Lagrange multiplier
        if (avg_density1 < self.avg_density_0):
            lagmin = float(self.lagrange)
            while (avg_density < self.avg_density_0):
                self.lagrange.assign(Constant(self.lagrange/2))
                avg_density = self.update_theta()
            lagmax = float(self.lagrange)
        elif (avg_density1 > self.avg_density_0):
            lagmax = float(self.lagrange)
            while (avg_density > self.avg_density_0):
                self.lagrange.assign(Constant(self.lagrange*2))
                avg_density = self.update_theta()
            lagmin = float(self.lagrange)
        else:
            lagmin = float(self.lagrange)
            lagmax = float(self.lagrange)

        # Dichotomy on Lagrange multiplier
        inddico=0
        while ((abs(1.-avg_density/self.avg_density_0)) > self.tol_mass):
            self.lagrange.assign(Constant((lagmax+lagmin)/2))
            avg_density = self.update_theta()
            inddico += 1;
            if (avg_density < self.avg_density_0):
                lagmin = float(self.lagrange)
            else:
                lagmax = float(self.lagrange)
        # print("   Dichotomy iterations:", inddico)


    def update_exponent(self, exponent_counter):
        exponent_counter += 1
        if (self.i < self.niternp):
            self.p.assign(Constant(1))
        elif (self.i >= self.niternp):
            if self.i == self.niternp:
                print("\n Starting penalized iterations\n")
            if ((abs(self.compliance-self.old_compliance) < 0.01*self.compliance_history[0]) and
                (exponent_counter > self.exponent_update_frequency) ):
                # average gray level
                gray_level = assemble((self.theta-self.thetamin)*(1.-self.theta)*dx)*4/self.volume
                self.p.assign(Constant(min(float(self.p)*(1+0.3**(1.+gray_level/2)), self.pmax)))
                exponent_counter = 0
                print("   Updated SIMP exponent p = ", float(self.p))
        return exponent_counter

