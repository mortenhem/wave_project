#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt


def solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T, 
		   user_action=None, version='scalar',
		   stability_safety_factor=0.9):


	
	
	x = np.linspace(0, Lx, Nx+1)
	y = np.linspace(0, Ly, Ny+1)

	dx = x[1] - x[0]
	dy = y[1] - y[0]
	
	xv = x[:, np.newaxis]
	yv = y[:, np.newaxis]

	if isinstance(q, (float, int)):
		q_max = q

	elif callable(q):
		# Find maximal q value
		q_values = []
		for i in range(0, len(x)):
			for j in range(0, len(y)):
				q_values.append(q(x[i], y[j]))

		# Pick out largest value
		q_max = max(q_values)

	
	stability_limit = (1./(np.sqrt(q_max)))*(1./np.sqrt(1./dx** + 1./dy**2))
	if dt <= 0:
		dt = stability_safety_factor*stability_limit
	
	elif dt > stability_limit:
		print 'Warning! dt exceeds the stability limit.'
		sys.exit()
	


	Nt = int(round(T/float(dt)))
	t = np.linspace(0, Nt*dt, Nt+1)
	Cx2  = (dt/dx)**2; Cy2 = (dt/dy)**2
	dt2 = dt*dt

	if f is None or f == 0:
		f = lambda x, y, t: 0 if version == 'scalar' else \
			lambda x, y, t: np.zeros((x.shape[0], y.shape[0]))

	if V is None or V == 0:
		V = lambda x, y: 0 if version == 'scalar' else \
			lambda x, y: np.zeros((x.shape[0], y.shape[1]))

	if I is None or I == 0:
		I = lambda x, y: 0 if version == 'scalar' else \
			lambda x, y: np.zeros((x.shape[0], y.shape[1]))


	u = np.zeros((Nx+1, Ny+1))
	u_1 = np.zeros((Nx+1, Ny+1))
	u_2 = np.zeros((Nx+1, Ny+1))
	f_a = np.zeros((Nx+1, Ny+1))

	Ix = range(0, u.shape[0])
	Iy = range(0, u.shape[1])
	It = range(0, t.shape[0])

	import time; t0 = time.clock()

	# Load initial condition into u_1
	if version == 'scalar':
		for i in range(0, Nx+1):
			for j in range(0, Ny+1):
				u_1[i,j] = I(x[i], y[j])

	else:
		None	# Use vectorized version

	
	if user_action is not None:
		user_action(u_1, x, xv, y, yv, t, 0)


	# Special formula for the first step
	n = 0
		
	if version == 'scalar':
		for i in Ix[1:-1]:
			for j in Iy[1:-1]:
				A = (q(x[i],y[j]) + q(x[i+1], y[j]))*(u_1[i+1,j] - u_1[i,j]) - \
					(q(x[i], y[j]) + q(x[i-1], y[j]))*(u_1[i,j] - u_1[i-1,j])

				B = (q(x[i],y[j]) + q(x[i], y[j+1]))*(u_1[i,j+1] - u_1[i,j]) - \
					(q(x[i], y[j]) + q(x[i], y[j-1]))*(u_1[i,j] - u_1[i,j-1])
	
				# Formula for the first time step, n=0		
				u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
						 0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)

	else:
		None	# Use vectorized version


	

#========Apply Neumann boundary conditions for first step=================
	
	# x=0
	i = Ix[0]
	ip1 = i+1
	im1 = ip1
		
	for j in Iy[1:-1]:
		A = (q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u[i,j]) - \
			(q(x[i], y[j]) + q(x[im1], y[j]))*(u_1[i,j] - u_1[im1,j])

		B = (q(x[i],y[j]) + q(x[i], y[j+1]))*(u_1[i,j+1] - u_1[i,j]) - \
			(q(x[i], y[j]) + q(x[i], y[j-1]))*(u_1[i,j] - u_1[i,j-1])
	
		u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
				0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)


	# x=Lx
	i = Ix[-1]
	ip1 = i-1
	im1 = ip1
	
	for j in Iy[1:-1]:
		A = (q(x[i],y[j]) + q(x[ip1], y[j]))*(u_1[ip1,j] - u[i,j]) - \
			(q(x[i], y[j]) + q(x[im1], y[j]))*(u_1[i,j] - u_1[im1,j])

		B = (q(x[i],y[j]) + q(x[i], y[j+1]))*(u_1[i,j+1] - u_1[i,j]) - \
			(q(x[i], y[j]) + q(x[i], y[j-1]))*(u_1[i,j] - u_1[i,j-1])
	
		u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
				0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)


	# y=0
	j = Iy[0]
	jp1 = j+1
	jm1 = jp1
	
	for i in Ix[1:-1]:
		A = (q(x[i],y[j]) + q(x[i+1], y[j]))*(u_1[i+1,j] - u_1[i,j]) - \
			(q(x[i], y[j]) + q(x[i-1], y[j]))*(u_1[i,j] - u_1[i-1,j])

		B = (q(x[i],y[j]) + q(x[i], y[jp1]))*(u_1[i,jp1] - u_1[i,j]) - \
			(q(x[i], y[j]) + q(x[i], y[jm1]))*(u_1[i,j] - u_1[i,jm1])
	
		u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
				0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)


	# y=Ly
	j = Iy[-1]
	jp1 = j-1	
	jm1 = jp1
	
	for i in Ix[1:-1]:
		A = (q(x[i],y[j]) + q(x[i+1], y[j]))*(u_1[i+1,j] - u_1[i,j]) - \
			(q(x[i], y[j]) + q(x[i+1], y[j]))*(u_1[i,j] - u_1[i-1,j])

		B = (q(x[i],y[j]) + q(x[i], y[jp1]))*(u_1[i,jp1] - u_1[i,j]) - \
			(q(x[i], y[j]) + q(x[i], y[jm1]))*(u_1[i,j] - u_1[i,jm1])
	
		u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
				0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)

#============================================================================

	if user_action is not None:
		user_action(u, x, xv, y, yv, t, 1)

	
	u_2, u_1, u = u_1, u, u_2	


	# Update inner mesh points
	for n in It[1:-1]:
		if version == 'scalar':
			for i in Ix[1:-1]:
				for j in Iy[1:-1]:
					
					A = (q(x[i],y[j]) + q(x[i+1], y[j]))*(u_1[i+1,j] - u_1[i,j]) - \
						(q(x[i], y[j]) + q(x[i-1], y[j]))*(u_1[i,j] - u_1[i-1,j])

					B = (q(x[i],y[j]) + q(x[i], y[j+1]))*(u_1[i,j+1] - u_1[i,j]) - \
						(q(x[i], y[j]) + q(x[i], y[j-1]))*(u_1[i,j] - u_1[i,j-1])

					u[i,j] = 2*u_1[i,j] - u_2[i,j]*(1 - b*dt/2.) + \
							 0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], t[n])



	
#========Apply Neumann boundary conditions for n>=0=================
	
		# x=0
		i = Ix[0]
		ip1 = i+1
		im1 = ip1
		
		for j in Iy[1:-1]:
			A = (q(x[i],y[j]) + q(x[ip1],y[j]))*(u_1[ip1,j] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[im1], y[j]))*(u_1[i,j] - u_1[im1,j])

			B = (q(x[i],y[j]) + q(x[i], y[j+1]))*(u_1[i,j+1] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[i], y[j-1]))*(u_1[i,j] - u_1[i,j-1])
	
			u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
					0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)

		# x=Lx
		i = Ix[-1]
		ip1 = i-1	
		im1 = ip1
				
		for j in Iy[1:-1]:			
			A = (q(x[i],y[j]) + q(x[ip1], y[j]))*(u_1[ip1,j] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[im1], y[j]))*(u_1[i,j] - u_1[im1,j])

			B = (q(x[i],y[j]) + q(x[i], y[j+1]))*(u_1[i,j+1] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[i], y[j-1]))*(u_1[i,j] - u_1[i,j-1])
	
			u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
					0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)

		# y=0
		j = Iy[0]
		jp1 = j+1
		jm1 = jp1
	
		for i in Ix[1:-1]:
			A = (q(x[i],y[j]) + q(x[i+1], y[j]))*(u_1[i+1,j] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[i-1], y[j]))*(u_1[i,j] - u_1[i-1,j])

			B = (q(x[i],y[j]) + q(x[i], y[jp1]))*(u_1[i,jp1] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[i], y[jm1]))*(u_1[i,j] - u_1[i,jm1])
	
			u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
					0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)

		# y=Ly
		j = Iy[-1]
		jp1 = j-1	
		jm1 = jp1
		
		for i in Ix[1:-1]:
	
			A = (q(x[i],y[j]) + q(x[i+1], y[j]))*(u_1[i+1,j] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[i+1], y[j]))*(u_1[i,j] - u_1[i-1,j])
	
			B = (q(x[i],y[j]) + q(x[i], y[jp1]))*(u_1[i,jp1] - u_1[i,j]) - \
				(q(x[i], y[j]) + q(x[i], y[jm1]))*(u_1[i,j] - u_1[i,jm1])
	
			u[i,j] = u_1[i,j] + dt*V(x[i], y[j]) - 0.5*b*dt2*V(x[i],y[j]) + \
					0.5*Cx2*A + 0.5*Cy2*B + dt2*f(x[i], y[j], n)

#============================================================================

		if user_action is not None:
			if user_action(u, x, xv, y, yv, t, n+1):
				break		
	

	
		# Update data structures before next step
		u_2, u_1, u = u_1, u, u_2

		
	u = u_1
	t1 = time.clock()
	cpu_time = t1 - t0
	
	return u, x, y, t, cpu_time




def test_constant_solution():
	
	def u_exact(x, y, t):
		return c

	def I(x, y):
		return u_exact(x, y, 0)

	def V(x, y):
		return 0.0

	def f(x, y, t):
		return 0.0

	
	# wave speed can be arbitrary
	def q(x, y):
		return x*(1-y) + y*(1+x)

	c = 2.5
	b = 0.5
	Lx = 10
	Ly = 15
	Nx = 10
	Ny = 10
	dt = 0.05
	T = 5
	
	def assert_no_error():
		u_e = u_exact(x, y, t[n])
		diff = np.abs(u - u_e).max()
		print diff
		tol = 1E-14
		assert diff < tol
		
	solver(I, V, f, q, b, Lx, Ly, Nx, Ny, dt, T, 
		   user_action=assert_no_error(), version='scalar',
		   stability_safety_factor=0.9)


if __name__=='__main__':

	#test_constant_solution()

