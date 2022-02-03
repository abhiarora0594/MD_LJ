import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import time


class MD:
	"""initializing the class"""
	def __init__(self, N, x0, v0, dt, t_initial, t_final, step_no, r_cut, T_des, tau_damp, alg0, L):
		self.x0 = x0
		self.v0 = v0
		self.N = N
		self.dt = dt
		self.t_initial = t_initial
		self.t_final = t_final	
		self.step = step_no
		self.r_cut = r_cut
		self.T_des = T_des
		self.tau_damp = tau_damp
		self.alg0 = alg0
		self.L = L

	
	def run_MD(self):
		
		self.initialize_variables()
		self.data_collectors()		
		self.velocity_verlet()
		self.plot_figures()

	
	def initialize_variables(self):
		self.t_sim = self.t_initial
		self.U = 0.0
		self.K = 0.0
		self.V = self.L**3
		self.P_virial = 0.0
		self.momtm = np.array([])
		self.com = np.array([])
		self.xp = self.x0 # position t
		self.vp = self.v0 # velocity t
		self.vhalf = np.zeros((self.N,3)) # velocity at t + delta t /2
		self.xn = np.zeros((self.N,3)) # position t+\delta t
		self.xpbc = np.zeros((self.N,3)) # before apply pbc position t+\delta t
		self.vn = np.zeros((self.N,3)) # velocity t+\delta t
		self.Fp = np.zeros((self.N,3)) # forces at t
		self.Fn = np.zeros((self.N,3)) # forces at t+\delta t
		self.zetap = 0.0 # zeta at t
		self.zetan = 0.0 # zeta at t + \delta t
		self.temp = 0.0
		self.pres = 0.0
		self.cumtemp = 0.0
		self.diffusion = 0.0
		self.msd = 0.0
		self.xequi = self.x0

	
	def data_collectors(self):
		self.Uall = np.array([])
		self.Kall = np.array([])
		self.tall = np.array([])
		self.momtmall = np.empty((0,3))
		self.comall = np.empty((0,3))
		self.tempall = np.array([])
		self.presall = np.array([])
		self.cumtempall = np.array([])
		self.msdall = np.array([])
		self.diffusionall = np.array([])

	
	def algorithm_NVE(self):

		self.vhalf = self.get_velocity(self.vp,self.Fp,self.dt)
		self.xpbc = self.get_position(self.xp,self.vhalf,self.dt)
		self.xn = self.apply_pbc(self.xpbc)
		self.Fn, self.U = self.get_force_and_potential(self.xn)
		self.vn = self.get_velocity(self.vhalf,self.Fn,self.dt)

		self.K = self.get_kinetic_energy(self.vn)
		self.momtm = self.get_momentum(self.vn)
		self.com = self.get_center_of_mass(self.xn)
		self.temp = self.get_temperatue()
		self.pres = self.get_pressure()
		self.msd, self.diffusion = self.get_mean_squared_displacement(self.xpbc)


	def algorithm_NVT(self):

		self.vhalf = self.get_velocity_nvt_half(self.vp,self.Fp,self.zetap,self.dt)
		self.xpbc = self.get_position(self.xp,self.vhalf,self.dt)
		self.xn = self.apply_pbc(self.xpbc)
		self.Fn, self.U = self.get_force_and_potential(self.xn)
		self.zetan = self.get_zeta(self.zetap,self.dt)
		self.vn = self.get_velocity_nvt_full(self.vhalf,self.Fn,self.zetan,self.dt)

		self.K = self.get_kinetic_energy(self.vn)
		self.momtm = self.get_momentum(self.vn)
		self.com = self.get_center_of_mass(self.xn)
		self.temp = self.get_temperatue()
		self.pres = self.get_pressure()	
		self.msd, self.diffusion = self.get_mean_squared_displacement(self.xpbc)

	
	def write_positions(self,f):

		f.write(str(N) + '\n')
		f.write('t = ' + str(self.t_sim) + '\n')
		for i in range(self.N):
			f.write('C' + ' ' + str(self.xp[i,0]) + ' ' +  
				str(self.xp[i,1]) + ' ' +  str(self.xp[i,2]) + '\n')

		
	def velocity_verlet(self):

		# fields at t = t_initial
		# print(self.L)
		start = time.time()

		# self.msd,self.diffusion = self.get_mean_squared_displacement(self.xp)
		self.xp = self.apply_pbc(self.xp)
		self.Fp, self.U = self.get_force_and_potential(self.xp)
		self.K = self.get_kinetic_energy(self.vp)
		self.momtm = self.get_momentum(self.vp)
		self.com = self.get_center_of_mass(self.xp)
		self.temp = self.get_temperatue()
		self.pres = self.get_pressure()


		# collecting fields for plots
		self.Uall = np.append(self.Uall,self.U)
		self.Kall = np.append(self.Kall,self.K)
		self.tall = np.append(self.tall,self.t_sim)
		self.momtmall = np.vstack((self.momtmall,self.momtm))
		self.comall = np.vstack((self.comall,self.com))
		self.tempall = np.append(self.tempall,self.temp)
		self.presall = np.append(self.presall,self.pres)
		self.cumtemp = self.temp
		self.cumtempall = np.append(self.cumtempall,self.cumtemp)

		print('step no is ' + str(self.step) + ' and time is ' + str(self.t_sim))
		print('U is ' + str(self.U) + ' and K is ' + str(self.K) + ' and total is ' + str(self.U + self.K)  + '\n')

		f = open("256.xyz", "w")
		f2 = open("equi.txt", "w")
		self.write_positions(f)

		count = 0
		tol = 1.0

		while (self.t_sim < self.t_final):

			# algorithm
			self.t_sim = self.t_sim + self.dt
			self.step = self.step + 1

			if (self.alg0 == 0):
				self.algorithm_NVE()
			elif (self.alg0 == 1):
				self.algorithm_NVT()				

			# updating time, steps and fields for next time step
			self.update_fields()

			self.cumtemp = (self.cumtemp*self.step + self.temp)/(self.step + 1) 
			
			# collecting fields for plots
			self.Uall = np.append(self.Uall,self.U)
			self.Kall = np.append(self.Kall,self.K)
			self.tall = np.append(self.tall,self.t_sim)
			self.momtmall = np.vstack((self.momtmall,self.momtm))
			self.comall = np.vstack((self.comall,self.com))
			self.tempall = np.append(self.tempall,self.temp)
			self.presall = np.append(self.presall,self.pres)
			self.cumtempall = np.append(self.cumtempall,self.cumtemp)
			self.msdall = np.append(self.msdall,self.msd)
			self.diffusionall = np.append(self.diffusionall,self.diffusion)

			tol = abs((self.cumtempall[self.step] - self.T_des)/self.T_des)

			if (tol < 1e-8):
				count = count + 1
			
			print_check = 0

			if (count == 10 and print_check == 0):
				np.savetxt('displacement_equi.txt',self.xn)
				np.savetxt('velocity_equi.txt',self.vn)
				f2.write(str(self.step) + '\n')
				f2.write('t = ' + str(self.t_sim) + '\n')
				print_check = 1	



			if (self.step%100 == 0):
				self.write_positions(f)

			if (self.step%5 == 0):
				print('step no is ' + str(self.step) + ' and time is ' + str(self.t_sim))
				print('U is ' + str(self.U) + ' and K is ' + str(self.K) + ' and total is ' + str(self.U + self.K))
				print('T is ' + str(self.temp) + ' and P is ' + str(self.pres) + '\n')

		print('vx at t= ' + str(self.t_sim) + ' is ' + str(self.momtmall[self.step,0]) + ' and vy is ' 
						+ str(self.momtmall[self.step,1]) + ' and vz is ' + str(self.momtmall[self.step,2]))

		print('x at t= ' + str(self.tall[0]) + ' is ' + str(self.comall[0,0]) + ' and y is ' 
						+ str(self.comall[0,1]) + ' and z is ' + str(self.comall[0,2]))

		print('x at t= ' + str(self.t_sim) + ' is ' + str(self.comall[self.step,0]) + ' and y is ' 
						+ str(self.comall[self.step,1]) + ' and z is ' + str(self.comall[self.step,2]))

		f.close()
		f2.close()		

		end = time.time()
		print(end - start)


	
	def get_force_and_potential(self,x):

		F = np.zeros((self.N,3))

		U = 0.0

		self.P_virial = 0.0

		L_half = 0.5*self.L

		t_c = (1/self.r_cut)**6

		F_c = (48*t_c**2 - 24*t_c)/self.r_cut
		U_c = 4*(t_c**2 - t_c)

		for i in range(self.N):
			for j in range(i+1,self.N):

				rx = x[i,0]-x[j,0]
				ry = x[i,1]-x[j,1]
				rz = x[i,2]-x[j,2]

				if (rx < -L_half):
					rx = rx + self.L
				elif (rx > L_half):
					rx = rx - self.L

				if (ry < -L_half):
					ry = ry + self.L
				elif (ry > L_half):
					ry = ry - self.L
					
				if (rz < -L_half):
					rz = rz + self.L
				elif (rz > L_half):
					rz = rz - self.L

				rij = np.sqrt(rx**2 + ry**2 + rz**2)		

				if (rij <= self.r_cut):

					t = (1/rij)**6
					prefac = (48*t**2 - 24*t)/rij - F_c
					prefac2 = 4*(t**2 - t) - U_c + (rij-self.r_cut)*F_c

					F[i,0] = F[i,0] + prefac*rx/rij
					F[i,1] = F[i,1] + prefac*ry/rij
					F[i,2] = F[i,2] + prefac*rz/rij

					self.P_virial = self.P_virial + prefac*rij

					if (j > i):
						F[j,0] = F[j,0] - prefac*rx/rij
						F[j,1] = F[j,1] - prefac*ry/rij
						F[j,2] = F[j,2] - prefac*rz/rij
	
					U = U + prefac2

		return F,U;

	
	def get_momentum(self,v):

		m = np.sum(v,axis=0)
		m2 = m.reshape(1,-1)
		# print(m2)

		return m2;

	
	def get_center_of_mass(self,x):

		m = np.sum(x,axis=0)
		m2 = m.reshape(1,-1)/self.N
		# print(m2)

		return m2;

	
	def get_kinetic_energy(self,v):

		K = 0.0

		# K = 0.5*np.sum(np.sum(np.multiply(v,v),axis=1))
		for i in range(self.N):
			K = K + 0.5*(v[i,0]**2 + v[i,1]**2 + v[i,2]**2)

		return K 

	
	def get_velocity(self,v,F,dt):

		vn = v + F*(dt/2);

		return vn;

	def get_velocity_nvt_half(self,v,F,zeta,dt):

		vn = v + (F - zeta*v)*(dt/2);

		return vn;

	def get_velocity_nvt_full(self,v,F,zeta,dt):

		vn = (v + F*(dt/2))/(1+zeta*(dt/2))	

		return vn;

	def get_zeta(self,zeta,dt):

		zetan = zeta + (1/self.tau_damp**2)*(self.temp/self.T_des-1)*dt

		return zetan;

	
	def get_position(self,x,v,dt):

		xn = x + v*dt;

		return xn;

	
	def update_fields(self):

		self.Fp = self.Fn
		self.xp = self.xn
		self.vp = self.vn
		self.zetap = self.zetan

	
	def get_temperatue(self):

		T = 2*self.K/(3*(self.N-1)) # non-dimensional temperature

		return T;

	
	def get_pressure(self):

		P = self.N*self.temp/self.V + 1/(3*self.V)*self.P_virial

		return P;

	def get_mean_squared_displacement(self,x):

		msd = 0

		for i in range(self.N):
			msd =  msd + (x[i,0]-self.xequi[i,0])**2 + (x[i,1]-self.xequi[i,1])**2 + (x[i,2]-self.xequi[i,2])**2

		msd = msd/self.N

		diffusion = msd/6.0/self.t_sim


		return msd, diffusion;

	
	def apply_pbc(self,x):

		xn = x

		# print(np.shape(xn))

		for i in range(self.N):

			if xn[i,0] < 0.0:
				xn[i,0] = xn[i,0] + self.L
			elif xn[i,0] > self.L:
				xn[i,0] = xn[i,0] - self.L

			if xn[i,1] < 0.0:
				xn[i,1] = xn[i,1] + self.L
			elif xn[i,1] > self.L:
				xn[i,1] = xn[i,1] - self.L

			if xn[i,2] < 0.0:
				xn[i,2] = xn[i,2] + self.L
			elif xn[i,2] > self.L:
				xn[i,2] = xn[i,2] - self.L		


		return xn;

	
	def plot_figures(self):

		plt.plot(self.tall,self.Uall,'-r',label = 'U(t)')
		plt.plot(self.tall,self.Kall,'-c',label = 'K(t)')
		plt.plot(self.tall,self.Uall+self.Kall,'-g',label = 'H(t)')
		plt.xlabel('time (t)')
		plt.ylabel('Energies')
		plt.title('Plot of various energies with time')
		plt.legend()
		# plt.show()

		plt.plot(self.tall,self.momtmall[:,0],'-r',label = 'p_x (t)')
		plt.plot(self.tall,self.momtmall[:,1],'-c',label = 'p_y (t)')
		plt.plot(self.tall,self.momtmall[:,2],'-g',label = 'p_z (t)')
		plt.xlabel('time (t)')
		plt.ylabel('Momentum in 3 direction')
		plt.title('Plot of Momentum with time')
		plt.legend()
		# plt.show()

		plt.plot(self.tall,self.comall[:,0],'-r',label = 'x_c(t)')
		plt.plot(self.tall,self.comall[:,1],'-c',label = 'y_c(t)')
		plt.plot(self.tall,self.comall[:,2],'-g',label = 'z_c(t)')
		plt.xlabel('time (t)')
		plt.ylabel('Center of mass position')
		plt.title('Plot of center of mass position with time')
		plt.legend()
		# plt.show()

		plt.plot(self.tall,self.tempall,'-r',label = 'T (t)')
		plt.xlabel('time (t)')
		plt.ylabel('Temperature')
		plt.title('Plot of temperature with time')
		plt.legend()
		# plt.show()

		plt.plot(self.tall,self.presall,'-r',label = 'P (t)')
		plt.xlabel('time (t)')
		plt.ylabel('Pressure')
		plt.title('Plot of pressure with time')
		plt.legend()
		# plt.show()

		plt.plot(self.tall[1:],self.msdall,'-r',label = 'MSD (t)')
		plt.xlabel('time (t)')
		plt.ylabel('Pressure')
		plt.title('Plot of mean squared displacement with time')
		plt.legend()
		# plt.show()

		plt.plot(self.tall[1:],self.diffusionall,'-r',label = 'D (t)')
		plt.xlabel('time (t)')
		plt.ylabel('Pressure')
		plt.title('Plot of diffusion with time')
		plt.legend()
		# plt.show()

		np.savetxt('potential_f.txt',self.Uall)
		np.savetxt('kinetic_f.txt',self.Kall)
		np.savetxt('momentum_f.txt',self.momtmall)
		np.savetxt('temperature_f.txt',self.tempall)
		np.savetxt('pressure_f.txt',self.presall)
		np.savetxt('cum_temp_f.txt',self.cumtempall)
		np.savetxt('com_f.txt',self.comall)
		np.savetxt('msd_f.txt',self.msdall)
		np.savetxt('diffusion_f.txt',self.diffusionall)
		np.savetxt('displacement_f.txt',self.xn)
		np.savetxt('velocity_f.txt',self.vn)



if __name__ == "__main__":

	# initial conditions
	x0 = np.loadtxt('liquid256.txt')
	N = np.shape(x0)[0]

	v0 = np.zeros((N,3))
	np.random.seed(20)

	v0[:,0] = 1.0*np.random.normal(0.0,0.8,N)
	v0[:,1] = 1.0*np.random.normal(0.0,0.8,N)
	v0[:,2] = 1.0*np.random.normal(0.0,0.8,N)

	v0[N-1,0] = v0[N-1,0] - np.sum(v0,axis=0)[0]
	v0[N-1,1] = v0[N-1,1] - np.sum(v0,axis=0)[1]
	v0[N-1,2] = v0[N-1,2] - np.sum(v0,axis=0)[2]	

	# check for total momentum is zero
	print(np.sum(v0,axis=0))

	# local variables
	dt = 0.002
	t_final = 100.0
	t_initial = 0.0
	step_no = 0
	r_cut = 2.5
	tau_damp = 0.05
	alg0 = 1 # 0 for NVE and 1 for NVT
	T_des = 0.831716
	L = 6.8

	# defining a class object
	objMD = MD(N,x0,v0,dt,t_initial,t_final,step_no,r_cut,T_des,tau_damp,alg0,L)

	# running the MD
	objMD.run_MD()





		
