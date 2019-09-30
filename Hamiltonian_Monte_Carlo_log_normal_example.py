#import all libraries and modules
import time
import copy
import os

import autograd.numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import pickle 
import scipy.linalg
import math
import scipy.stats as st
from scipy.stats import norm
from collections import deque
from autograd import grad



##########################################################################################		
#---------------------------------FUNCTIONS----------------------------------------------#
##########################################################################################	


#-----------------------------------------------------------------------------------------	

def potential_energy_normal_distribution(mu,sigma):
	""" Input
	-----------
	logp(x | mu, sigma) = 0.5 * log(2pi) + log(sigma) + 0.5 * ((x - mu)/sigma)^2
	mu    : mean, float
	sigma : standard deviation, float
	
	Output
	-----------
	U : potential energy, float
	
	"""	
	def logp(x):
		return 0.5 * (np.log(2 * np.pi * sigma * sigma) + ((x - mu)/sigma) **2)
		
	
	return logp

#-----------------------------------------------------------------------------------------

def leapfrogIntegrator(q, p, dUdq, path_len, step_size):
	"""Inputs
	-----------
	q         : Initial position, np.float
	p         : Initial momentum, np.float
	dUdq      : callable function for gradient of potential energy
	path_len  : integration time L, float
	step_size : how long each integration should be, float
	
	Outputs
	-----------
	q,p : new position and momentum, floats
	"""
	
	q, p = np.copy(q), np.copy(p)
	
	#make a 1st half step for momentum
	p -= step_size * dUdq(q)/2.0
	
	#Alternate full steps for position and momentum except at the end of the trajectory
	for _ in np.arange(np.round(path_len / step_size) -1):
		q += step_size * p 
		p -= step_size * dUdq(q)
	
	#Last full step for position and half step for momentum	
	q += step_size * p 
	p -= step_size * dUdq(q)/2.0
	
	#momentum flip at the end
	return q, -p 
	
#-----------------------------------------------------------------------------------------

def leapfrogIntegrator_slow(q, p, dUdq, path_len, step_size):
	"""
	This is slower due to saving all positions and momentums in integration
	
	Inputs
	-----------
	q         : Initial position, np.float
	p         : Initial momentum, np.float
	dUdq      : callable function for gradient of potential energy
	path_len  : integration time L, float
	step_size : how long each integration should be, float
	
	Outputs
	-----------
	q,p : new position and momentum, floats
	"""
	
	q, p = np.copy(q), np.copy(p)
	
	positions, momentums = [np.copy(q)], [np.copy(p)] # Collect them in a list
	
	#make a 1st half step for momentum
	p -= step_size * dUdq(q)/2.0
	
	#Alternate full steps for position and momentum except at the end of the trajectory
	for _ in np.arange(np.round(path_len / step_size) -1):
		q += step_size * p 
		p -= step_size * dUdq(q)
		
		#Save
		positions.append(np.copy(q))
		momentums.append(np.copy(p)) 
	
	#Last full step for position and half step for momentum	
	q += step_size * p	
	p -= step_size * dUdq(q)/2.0
	
	#Save
	positions.append(q)
	momentums.append(p)
	
	#momentum flip at the end
	return q, -p, np.array(positions), np.array(momentums)
	
#-----------------------------------------------------------------------------------------

def hamiltonianMonteCarlo_sampling(n_samples, potential_energy, initial_position, path_len = 1.0, step_size = 0.5):
	""" Inputs
	-----------
	n_samples         : number of samples to return, integer
	potential_energy  : callable function of potential energy
	initial_position  : initial position for sampling, np.array
	path_len          : how long each integration path is ; smaller is faster and more correlated, float
	step_size         : initial step size for the leapfrog integration, which will be tuned, float
	
	Outputs
	-----------
	np.array of n_samples
	"""
	
	initial_position = np.array(initial_position)
	samples          = [initial_position] #collect in a list
	
	momentum = st.norm(0,1) #draw the momentum
	
	dUdq = grad(potential_energy) #gradient function for potential energy 
	
	#! If the initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws we can do this in one call to np.random.normal, and iterate over rows
	size = (n_samples,) + initial_position.shape[:1]
	for p0 in momentum.rvs(size=size):
		#Integrate over path to get a new position and momentum
		q_new, p_new = leapfrogIntegrator(samples[-1],p0, dUdq, path_len = path_len, step_size = step_size)
				
	#Check for Metropolis acceptance criterion
	start_log_p = np.sum(momentum.logpdf(p0))    - potential_energy(samples[-1])
	new_log_p   = np.sum(momentum.logpdf(p_new)) - potential_energy(q_new)
	p_accept    = min(1.0, np.exp(new_log_p - start_log_p))
	if np.random.rand() < p_accept:
		samples.append(q_new)
	else:
		samples.append(np.copy(samples[-1]))
	
	return np.array(samples[1:])
	
#-----------------------------------------------------------------------------------------

def hamiltonianMonteCarlo_sampling_slow(n_samples, potential_energy, initial_position, path_len = 1., step_size = 0.5):
	""" Inputs
	-----------
	n_samples         : number of samples to return, integer
	potential_energy  : callable function of potential energy
	initial_position  : initial position for sampling, np.array
	path_len          : how long each integration path is ; smaller is faster and more correlated, float
	step_size         : initial step size for the leapfrog integration, which will be tuned, float
	
	Outputs
	-----------
	np.array of n_samples
	"""
	
	initial_position = np.array(initial_position)
	samples          = [initial_position] #collect in a list
	
	momentum = st.norm(0,1) #draw the momentum
	
	dUdq = grad(potential_energy) #gradient function for potential energy
	
	# Information to be stored
	sampled_positions = []
	sampled_momentums = []
	acceptance_hist   = [] 
	
	#! If the initial_position is a 10d vector and n_samples is 100, we want 100 x 10 momentum draws we can do this in one call to np.random.normal, and iterate over rows
	size = (n_samples,) + initial_position.shape[:1]
	for p0 in momentum.rvs(size=size):
		#Integrate over path to get a new position and momentum
		q_new, p_new, positions, momentums = leapfrogIntegrator_slow(samples[-1],p0, dUdq, path_len = 2 * np.random.rand() * path_len, step_size = step_size)
		
		# Save all positions and momentums
		sampled_positions.append(positions)
		sampled_momentums.append(momentums)
		
		#Check for Metropolis acceptance criterion
		start_log_p = np.sum(momentum.logpdf(p0))    - potential_energy(samples[-1])
		new_log_p   = np.sum(momentum.logpdf(p_new)) - potential_energy(q_new)
		p_accept    = min(1.0, np.exp(new_log_p - start_log_p))
		if np.random.rand() < p_accept:
			samples.append(q_new)
			acceptance_hist.append(1.0)
		else:
			samples.append(np.copy(samples[-1]))
			acceptance_hist.append(0.0)
	
	return (np.array(samples[1:]) , np.array(sampled_positions), np.array(sampled_momentums), np.array(acceptance_hist))

#-----------------------------------------------------------------------------------------

##########################################################################################		
#-----------------------------------EXAMPLE----------------------------------------------#
##########################################################################################


samples, positions, momentums, acceptance_hist = hamiltonianMonteCarlo_sampling_slow(50, potential_energy_normal_distribution(0.0,0.1), initial_position = 0.0, step_size = 0.01)

# Save them in a dictionary
results = dict()
results['samples']         = samples
results['positions']       = positions
results['momentums']       = momentums
results['acceptance_hist'] = acceptance_hist
spio.savemat('test.mat', results)




    
 
    
