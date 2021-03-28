import torch
import torch_fenics

from fenics import *
from fenics_adjoint import *
# from dolfin import *
from ufl import nabla_div

import logging

class Corroded(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        # return on_boundary and (between(x[0], (0.5, 0.7)) )
                        #y                          #x
        return (between(x[1], (0, 0.2)) and between(x[0], (0.5, 0.7)))


class Healthy(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and (between(x[0], (0., 0.5)) or between(x[0], (0.7, 1.)))


class K(Expression):
    # material
    def __init__(self, materials, k_0, k_1, **kwargs):
        self.materials = materials
        self.k_0 = k_0
        self.k_1 = k_1

    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        else:
            values[0] = self.k_1


class HomogeneousBeam(torch_fenics.FEniCSModule):
    # Construct variables which can be reused in the constructor
    def __init__(self):
        # Call super constructor
        super().__init__()
        #TODO: make L, W variables
        # Scaled variables
        self.bc = None
        self.L = 1
        self.W = 0.2

        self.rho = 1
        self.delta = self.W/self.L
        self.gamma = 0.4*self.delta**2
        self.g = self.gamma
        self.tol = 1e-14 # boundary condition

        # Create function space                                        # of cells in x-y-z
        self.mesh = BoxMesh(Point(0, 0, 0), Point(self.L, self.W, self.W), 10, 3, 3)
        self.V = VectorFunctionSpace(self.mesh, 'P', 1)


        # Inhomogeneity
        # try1: CellFunction is not imported!
        # self.materials = CellFunction('size_t', self.mesh)
        # subdomain_0 = CompiledSubDomain('x[0] <= 0.5 + tol || x[0] >= 0.7 - tol', tol=self.tol) # healthy
        # subdomain_1 = CompiledSubDomain('x[0] >= 0.5 - tol && x[0] <= 0.7 + tol', tol=self.tol) # corroded
        # subdomain_0.mark(materials, 0)
        # subdomain_1.mark(materials, 1)

        # self.k_0 = 0
        # self.k_1 = 1
        # self.kappa = K(self.sub_domains, self.k_0, self.k_1, degree=0)

        # try2: 
        # self.corroded = Corroded()
        # self.domains = MeshFunction("size_t", self.mesh, self.mesh.topology().dim())
        # self.domains.set_all(1)
        # self.corroded.mark(self.domains, 0)

        # self.a0 = Constant(0.01) # corroded
        # self.a1 = Constant(1.)   # healthy
        
        # dx = Measure('dx', domain=self.mesh, subdomain_data=self.domains)
    

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

        #try1:
        # L = self.kappa* dot(self.f, self.v)*dx + dot(self.T, self.v)*ds

        # try2:
        # L =  (inner(a0*grad(u), grad(v))*dx(0) + inner(a1*grad(u), grad(v))*dx(1)
        # L = 0.01 * dot(self.f, self.v)*dx(0) + 1.0 * dot(self.f, self.v)*dx(1) + dot(self.T, self.v)*ds

        # Construct boundary condition
        self.bc_l = DirichletBC(self.V, Constant((0, 0, 0)), self.clamped_boundary_left)
 


        # Solve
        self.u = Function(self.V) # displacement
        solve(self.a == L, self.u, self.bc_l)


        # Return the solution
        return self.u

    def input_templates(self):
        # Declare input shapess
        return Constant((1)), Constant((1)), Constant((1))

    def exec(self, expression):
        # To work with fenics' functions
        if 'return' in expression: return exec(expression.split(' ')[1])
        else: exec(expression)