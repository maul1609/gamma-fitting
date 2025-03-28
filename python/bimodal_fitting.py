import scipy.io as sio
import numpy as np
import hdf5storage
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.optimize import differential_evolution
from lmfit import Parameters, Minimizer
from lmfit.minimizer import MinimizerResult
from scipy.linalg import eigvals

def robust_roots(coeffs):
    coeffs = np.trim_zeros(coeffs, 'f')
    n = len(coeffs) - 1
    if n == 0:
        return np.array([])

    A = np.diag(np.ones(n - 1), -1)
    A[0, :] = -coeffs[1:] / coeffs[0]
    return eigvals(A)
    
    
import warnings
warnings.filterwarnings("ignore")

global y_data
global x_data
global M0fit
global M2fit
global M3fit
global M6fit
global c

# Example model: y = a * np.exp(-b * x) + c
def model(x,a,b,c1):
	# need M0,M2,M3,M6
	M0_1 = M0fit*a
	M2_1 = M2fit*b
	M3_1 = M3fit*c1
	M0_2 = M0fit*(1.0-a)
	M2_2 = M2fit*(1.0-b)
	M3_2 = M3fit*(1.0-c1)
	
	"""
		calculate the fit parameters for mode1 
	"""
	F=M2_1**3/np.maximum(M3_1**2*M0_1,1e-30)
	# find the roots
# 	print([(1.0-F[0]),(3.0-6.0*F[0]),(2.0-9.0*F[0])])
	mur1 = robust_roots(np.array([(1.0-F[0]),(3.0-6.0*F[0]),(2.0-9.0*F[0])]))
	# lambdas
	lambdas1 = gamma(mur1+4.0)/gamma(mur1+3.0)*M2_1/(M3_1)
	# calculate n0s
	n0s1 = M0_1*lambdas1**(mur1+1.0)/gamma(mur1+1.0)
	
	# moments calculated
	M0calc1=n0s1*gamma(mur1+1.0)/lambdas1**(mur1+1.0)
	M2calc1=n0s1*gamma(mur1+3.0)/lambdas1**(mur1+3.0)
	M3calc1=n0s1*gamma(mur1+4.0)/lambdas1**(mur1+4.0)
	M6calc1=n0s1*gamma(mur1+7.0)/lambdas1**(mur1+7.0)

	"""
		calculate the fit parameters for mode2
	"""
	F=M2_2**3/np.maximum(M3_2**2*M0_2,1e-30)
	# find the roots
# 	print([(1.0-F[0]),(3.0-6.0*F[0]),(2.0-9.0*F[0])])
	mur2 = robust_roots(np.array([(1.0-F[0]),(3.0-6.0*F[0]),(2.0-9.0*F[0])]))
	# lambdas
	lambdas2 = gamma(mur2+4.0)/gamma(mur2+3.0)*M2_2/(M3_2)
	# calculate n0s
	n0s2 = M0_2*lambdas2**(mur2+1)/gamma(mur2+1)
	
	# moments calculated
	M0calc2=n0s2*gamma(mur2+1.0)/lambdas2**(mur2+1.0)
	M2calc2=n0s2*gamma(mur2+3.0)/lambdas2**(mur2+3.0)
	M3calc2=n0s2*gamma(mur2+4.0)/lambdas2**(mur2+4.0)
	M6calc2=n0s2*gamma(mur2+7.0)/lambdas2**(mur2+7.0)

	"""
		now we need to add the different combinations together
		and pick the one that fits
	"""	
	M6calc11 = np.nanmax([M6calc1[0],0]) + np.nanmax([M6calc2[0],0])
	M6calc12 = np.nanmax([M6calc1[0],0]) + np.nanmax([M6calc2[1],0])
	M6calc21 = np.nanmax([M6calc1[1],0]) + np.nanmax([M6calc2[0],0])
	M6calc22 = np.nanmax([M6calc1[1],0]) + np.nanmax([M6calc2[1],0])

# 	is1=np.isfinite(n0s1)
# 	is2=np.isfinite(n0s2)
# 	n01=n0s1[is1][0]
# 	n02=n0s2[is1][0]
# 	mu1=mur1[is1][0]
# 	mu2=mur2[is1][0]
# 	lam1=lambdas1[is1][0]
# 	lam2=lambdas2[is1][0]	
# 	print(M0calc1,mur1,M2_1,M3_1)

	diffs = np.abs(np.array([M6fit-M6calc11,M6fit-M6calc12, \
		M6fit-M6calc21,M6fit-M6calc22]))
	
	ind,=np.where(diffs == np.min(diffs));ind=ind[0]
	
	if ind == 0:
		n01=n0s1[0]
		n02=n0s2[0]
		mu1=mur1[0]
		mu2=mur2[0]
		lam1=lambdas1[0]
		lam2=lambdas2[0]		
	elif ind == 1:
		n01=n0s1[0]
		n02=n0s2[1]
		mu1=mur1[0]
		mu2=mur2[1]
		lam1=lambdas1[0]
		lam2=lambdas2[1]
	elif ind == 2:
		n01=n0s1[1]
		n02=n0s2[0]
		mu1=mur1[1]
		mu2=mur2[0]
		lam1=lambdas1[1]
		lam2=lambdas2[0]
	elif ind == 3:
		n01=n0s1[1]
		n02=n0s2[1]	 
		mu1=mur1[1]
		mu2=mur2[1]
		lam1=lambdas1[1]
		lam2=lambdas2[1]
	
	return (np.maximum(n01,0)*x**mu1*np.exp(-np.maximum(lam1,0.)*x)+ \
    	np.maximum(n02,0)*x**mu2*np.exp(-np.maximum(lam2,0.)*x))*x**c['power']

# Objective function: sum of squared errors
def objective(params):
    a, b, c1 = params
    return np.sum((y_data - model(x_data, a, b, c1))**2)

def residuals(params, x, y):
	return model(params,x)-y

start1=324

dataload=hdf5storage.loadmat('../matlab/shitong.mat')

(r,c1)=np.shape(dataload['now_data_DP'])
dedge=np.insert(dataload['d_size'],0,2)



# store all the moments
M0 = dataload['now_total_Nd']*1e6
M2 = np.sum(dataload['now_data_DP']*(dataload['d_size']/1e6)**2,axis=1)*1e6
M3 = np.sum(dataload['now_data_DP']*(dataload['d_size']/1e6)**3,axis=1)*1e6
M6 = np.sum(dataload['now_data_DP']*(dataload['d_size']/1e6)**6,axis=1)*1e6

# arrays for parameters
mu1 = np.zeros(r)
mu2 = np.zeros(r)
lam1 = np.zeros(r)
lam2 = np.zeros(r)
n01 = np.zeros(r)
n02 = np.zeros(r)
M0_1 = np.zeros(r)
M0_2 = np.zeros(r)
M2_1 = np.zeros(r)
M2_2 = np.zeros(r)
M3_1 = np.zeros(r)
M3_2 = np.zeros(r)
M0calc = np.zeros(r)
M2calc = np.zeros(r)
M3calc = np.zeros(r)
M6calc = np.zeros(r)


# initial estimates and bounds
init1 = np.array([0.3,0.6,0.85])
minis = np.array([1.e-4,1.e-4,1.e-4])
maxis = 1.0-minis
Xstore = np.zeros((r,3))
# Bounds for a, b, c
bounds = [(minis[0], maxis[0]), \
	(minis[1],maxis[1]), (minis[2],maxis[2])]



c = dict()
c['power']=0
c['powerPlot']=0
c['M0']=M0
c['M2']=M2
c['M3']=M3
c['M6']=M6

"""
	define function to be fitted
"""
x_data = dataload['d_size'][0,:]/1e6
y_data = dataload['now_data_DP'][start1,:]*1e12/np.diff(dedge)*x_data**c['power']


"""
	simple single mode fit
	
	F=M(2)^3/(M(3)^2*M(0))
	solve
	(1-F)*mu^2+(3-6F)*mu+(2-9F)=0
	
	lam=gamma(mu+4)/gamma(mu+3)*M(2)/M(3)
	n0=M(0)*lam^(mu+1)/gamma(mu+1)

"""
for i in range(r):
	# Fcalc
	F=M2[i]**3/(M3[i]**2*M0[i])
	# find the roots
	mur = robust_roots(np.array([(1.-F[0]),(3.-6.*F[0]),(2.-9.*F[0])]))
	# lambdas
	lambdas = gamma(mur+4)/gamma(mur+3)*M2[i]/M3[i]
	# calculate n0s
	n0s = M0[i]*lambdas**(mur+1)/gamma(mur+1)
	
	# moments calculated
	M0calc1=n0s*gamma(mur+1)/lambdas**(mur+1)
	M2calc1=n0s*gamma(mur+3)/lambdas**(mur+3)
	M3calc1=n0s*gamma(mur+4)/lambdas**(mur+4)
	M6calc1=n0s*gamma(mur+7)/lambdas**(mur+7)

	# now choose the one that is the best fit
	diffs = np.abs(M6[i]-M6calc1)
	ind,=np.where(diffs == np.min(diffs));ind=ind[0]
	
	# reset them
	n01[i] = n0s[ind]
	lam1[i] = lambdas[ind]
	mu1[i] = mur[ind]

	M0calc[i]=n01[i]*gamma(mu1[i]+1)/lam1[i]**(mu1[i]+1)
	M2calc[i]=n01[i]*gamma(mu1[i]+3)/lam1[i]**(mu1[i]+3)
	M3calc[i]=n01[i]*gamma(mu1[i]+4)/lam1[i]**(mu1[i]+4)
	M6calc[i]=n01[i]*gamma(mu1[i]+7)/lam1[i]**(mu1[i]+7)

"""
 do the non-linear bimodal fit
"""
M0fit=M0[start1]
M2fit=M2[start1]
M3fit=M3[start1]
M6fit=M6[start1]

# Global optimization
result = differential_evolution(objective, bounds)

"""
params = Parameters()
params.add('a', min=minis[0], max=maxis[0])
params.add('b', min=minis[1], max=maxis[1])
params.add('c', min=minis[2], max=maxis[2])

# Set up Minimizer
mini = Minimizer(residuals, params, fcn_args=(x_data, y_data))

# Run brute-force global optimization
result = mini.minimize(method='brute', workers=1)  # parallel brute-force

# Print results
result.params.pretty_print()	
"""
plt.ion()
plt.plot(x_data,y_data)