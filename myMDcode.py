import numpy as np
import math
import matplotlib.pyplot as plt

class MD:
	"""initializing the class"""
	def __init__(self, N, x0, v0, dt, t_initial, t_final, step_no):
		self.x0 = x0
		self.v0 = v0
		self.N = N
		self.dt = dt
		self.t_initial = t_initial
		self.t_final = t_final	
		self.step = step_no

	def run_MD(self):

		self.initialize_variables()
		self.data_collectors()		
		self.velocity_verlet()
		self.plot_figures()

	def initialize_variables(self):
		self.t_sim = self.t_initial
		self.U = 0
		self.K = 0
		self.momtm = np.array([])
		self.com = np.array([])
		self.xp = self.x0 #position t
		self.vp = self.v0 #velocity t
		self.vhalf = np.zeros((self.N,3)) #velocity at t + delta t /2
		self.xn = np.zeros((self.N,3)) #position t+\delta t
		self.vn = np.zeros((self.N,3)) #velocity t+\delta t
		self.Fp = np.zeros((self.N,3)) #forces at t
		self.Fn = np.zeros((self.N,3)) #forces at t+\delta t

	def data_collectors(self):
		self.Uall = np.array([])
		self.Kall = np.array([])
		self.tall = np.array([])
		self.momtmall = np.empty((0,3))
		self.comall = np.empty((0,3))

	def algorithm_per_time_step(self):

		self.vhalf = self.get_velocity(self.vp,self.Fp,self.dt)
		self.xn = self.get_position(self.xp,self.vhalf,self.dt)
		self.Fn, self.U = self.get_force_and_potential(self.xn)
		self.vn = self.get_velocity(self.vhalf,self.Fn,self.dt)

		self.K = self.get_kinetic_energy(self.vn)
		self.momtm = self.get_momentum(self.vn)
		self.com = self.get_center_of_mass(self.xn)


	def write_positions(self,f):

		f.write(str(N) + '\n')
		f.write('t = ' + str(self.t_sim) + '\n')
		for i in range(self.N):
			f.write('C' + ' ' + str(self.xp[i,0]) + ' ' +  
				str(self.xp[i,1]) + ' ' +  str(self.xp[i,2]) + '\n')

	def velocity_verlet(self):

		# fields at t = t_initial
		self.Fp, self.U = self.get_force_and_potential(self.xp)
		self.K = self.get_kinetic_energy(self.vp)
		self.momtm = self.get_momentum(self.vp)
		self.com = self.get_center_of_mass(self.xp)

		# collecting fields for plots
		self.Uall = np.append(self.Uall,self.U)
		self.Kall = np.append(self.Kall,self.K)
		self.tall = np.append(self.tall,self.t_sim)
		self.momtmall = np.vstack((self.momtmall,self.momtm))
		self.comall = np.vstack((self.comall,self.com))

		print('step no is ' + str(self.step) + ' and time is ' + str(self.t_sim))
		print('U is ' + str(self.U) + ' and K is ' + str(self.K) + ' and total is ' + str(self.U + self.K)  + '\n')

		f = open("10.xyz", "w")
		self.write_positions(f)

		while (self.t_sim < self.t_final):

			# algorithm
			self.algorithm_per_time_step()

			# updating time, steps and fields for next time step
			self.t_sim = self.t_sim + self.dt
			self.step = self.step + 1
			self.update_fields()

			# collecting fields for plots
			self.Uall = np.append(self.Uall,self.U)
			self.Kall = np.append(self.Kall,self.K)
			self.tall = np.append(self.tall,self.t_sim)
			self.momtmall = np.vstack((self.momtmall,self.momtm))
			self.comall = np.vstack((self.comall,self.com))

			if (self.step%50 == 0):
				self.write_positions(f)

		print('step no is ' + str(self.step) + ' and time is ' + str(self.t_sim))
		print('U is ' + str(self.U) + ' and K is ' + str(self.K) + ' and total is ' + str(self.U + self.K)  + '\n')

		f.close()


	def get_force_and_potential(self,x):

		F = np.zeros((self.N,3))

		U = 0

		for i in range(self.N):
			for j in range(self.N):
				if (j != i):
					rij = np.sqrt((x[i,0]-x[j,0])**2 + (x[i,1]-x[j,1])**2 + (x[i,2]-x[j,2])**2) 

					prefac = 48*np.power((1/rij),13) - 24*np.power((1/rij),7)
					prefac2 = 4*(np.power((1/rij),12) - np.power((1/rij),6))

					F[i,0] = F[i,0] + prefac*(x[i,0]-x[j,0])/rij
					F[i,1] = F[i,1] + prefac*(x[i,1]-x[j,1])/rij
					F[i,2] = F[i,2] + prefac*(x[i,2]-x[j,2])/rij

					U = U + 0.5*prefac2

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

		K = 0

		for i in range(self.N):
			K = K + 0.5*(v[i,0]**2 + v[i,1]**2 + v[i,2]**2)

		return K 

	def get_velocity(self,v,F,dt):

		vn = v + F*dt/2;

		return vn;

	def get_position(self,x,v,dt):

		xn = x + v*dt;

		return xn;

	def update_fields(self):

		self.Fp = self.Fn
		self.xp = self.xn
		self.vp = self.vn

	def plot_figures(self):

		plt.plot(self.tall,self.Uall,'-r',label = 'U(t)')
		plt.plot(self.tall,self.Kall,'-c',label = 'K(t)')
		plt.plot(self.tall,self.Uall+self.Kall,'-g',label = 'H(t)')
		plt.legend()
		plt.show()

		plt.plot(self.tall,self.momtmall[:,0],'-r',label = 'v_x (t)')
		plt.plot(self.tall,self.momtmall[:,1],'-c',label = 'v_y (t)')
		plt.plot(self.tall,self.momtmall[:,2],'-g',label = 'v_z (t)')
		plt.legend()
		plt.show()

		plt.plot(self.tall,self.comall[:,0],'-r',label = 'x_c(t)')
		plt.plot(self.tall,self.comall[:,1],'-c',label = 'y_c(t)')
		plt.plot(self.tall,self.comall[:,2],'-g',label = 'z_c(t)')
		plt.legend()
		plt.show()


if __name__ == "__main__":

	#initial conditions
	x0 = np.loadtxt('10.txt')
	N = np.shape(x0)[0]
	v0 = np.zeros((N,3))

	#local variables
	dt = 0.002;
	t_final = 10.0;
	t_initial = 0.0;
	step_no = 0

	# defining a class object
	objMD = MD(N,x0,v0,dt,t_initial,t_final,step_no)

	# running the MD
	objMD.run_MD()

		