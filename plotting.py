import numpy as np
import matplotlib.pyplot as plt
import time

Uall = np.loadtxt('potential_f.txt')
Kall = np.loadtxt('kinetic_f.txt')
momtmall = np.loadtxt('momentum_f.txt')
tempall = np.loadtxt('temperature_f.txt')
presall = np.loadtxt('pressure_f.txt')
comall = np.loadtxt('com_f.txt')
msdall = np.loadtxt('msd_f.txt')
diffusionall = np.loadtxt('diffusion_f.txt')
cum_temp = np.loadtxt('cum_temp_f.txt')
tall =  np.linspace(0,100.002,50002)
print(np.shape(tall))
T_des = 0.831716

# cum_temp = []
# # cum_pres = []
# sum1 = 0
# # sum2 = 0

# for i in range(len(tempall)):

# 	sum1 = sum1 + tempall[i]
# 	# sum2 = sum2 + presall[i]
# 	temp1 = sum1/(i+1)
# 	# temp2 = sum2/(i+1)

# 	cum_temp = np.append(cum_temp,temp1)
# 	cum_pres = np.append(cum_pres,temp2)

# sum1 = 0
# sum2 = 0

# # for i in range(50001,len(tempall)):

# # 	if i < 50002: 
# # 		print(i)

# # 	sum1 = sum1 + tempall[i]
# # 	sum2 = sum2 + presall[i]

summ1 = sum(tempall)/len(tempall)
summ2 = sum(presall)/len(presall)
summ3 = sum(Uall)/len(Uall)
summ4 = sum(Kall)/len(Kall)
print(summ1)
print(summ2)
print(summ3)
print(summ4)


tol = 1.0
i = 0
count = 0 
while (count < 5):
	tol = abs((cum_temp[i+1] - T_des)/T_des)
	i = i + 1
	if (tol < 1e-8):
		count = count + 1

print(i)
print(tall[i])	

# tol = 1.0
# i = 0 
# count = 0
# while (count < 10):
# 	tol = abs((cum_pres[i+1] - cum_pres[i])/cum_pres[i])
# 	i = i + 1
# 	if (tol < 1e-8):
# 		count = count + 1
	
# print(i)
# print(tall[i])

plt.plot(tall,Uall,'-r',label = 'U(t)')
plt.plot(tall,Kall,'-c',label = 'K(t)')
plt.plot(tall,Uall+Kall,'-g',label = 'H(t)')
plt.xlabel('time (t)')
plt.ylabel('Energies')
plt.title('Plot of various energies with time')
plt.legend()
plt.show()

plt.plot(tall,momtmall[:,0],'-r',label = 'p_x (t)')
plt.plot(tall,momtmall[:,1],'-c',label = 'p_y (t)')
plt.plot(tall,momtmall[:,2],'-g',label = 'p_z (t)')
plt.xlabel('time (t)')
plt.ylabel('Momentum in 3 direction')
plt.title('Plot of Momentum with time')
plt.legend()
plt.show()

plt.plot(tall,comall[:,0],'-r',label = 'x_c(t)')
plt.plot(tall,comall[:,1],'-c',label = 'y_c(t)')
plt.plot(tall,comall[:,2],'-g',label = 'z_c(t)')
plt.xlabel('time (t)')
plt.ylabel('Center of mass position')
plt.title('Plot of center of mass position with time')
plt.legend()
plt.show()

plt.plot(tall,tempall,'-r',label = 'T (t)')
plt.xlabel('time (t)')
plt.ylabel('Temperature')
plt.title('Plot of temperature with time')
plt.legend()
plt.show()

plt.plot(tall,presall,'-r',label = 'P (t)')
plt.xlabel('time (t)')
plt.ylabel('Pressure')
plt.title('Plot of pressure with time')
plt.legend()
plt.show()

plt.plot(tall[1:],msdall,'-r',label = 'MSD (t)')
plt.xlabel('time (t)')
plt.ylabel('MSD')
plt.title('Plot of mean squared displacement with time')
plt.legend()
plt.show()

plt.plot(tall[1:],diffusionall,'-r',label = 'D (t)')
plt.xlabel('time (t)')
plt.ylabel('Diffusion')
plt.title('Plot of diffusion with time')
plt.legend()
plt.show()

plt.plot(tall,cum_temp,'-r',label = 'T_{avg} (t)')
plt.xlabel('time (t)')
plt.ylabel('Averaged Temperature')
plt.title('Plot of cummulative averaged temperature with time')
plt.legend()
plt.show()


# plt.plot(tall,cum_pres,'-r',label = 'P_{avg} (t)')
# plt.xlabel('time (t)')
# plt.ylabel('Averaged Pressure')
# plt.title('Plot of cummulative averaged pressure with time')
# plt.legend()
# plt.show()