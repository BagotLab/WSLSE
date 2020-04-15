import numpy as np 
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import seaborn as sns
from math import log
from scipy.stats import *
from scipy.stats.distributions import beta, gamma
from scipy.optimize import fmin
from scipy.optimize import brute
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sample_data_1Back
import sample_data_2Back
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from pylab import *

####################################  USE TO INITIALIZE MODEL SELECTION  ###########################################
obs_variables = "1Back" # change to "2Back" to use all four variables
paradigm = "No_PRL" # change to "No_PRL" if data is fixed contingency 80/20
trial_num = 100 # vs. 1000 Trials
#########################################################################################################################

def getFullPosterior1Back(ws,ls,means_wsr,means_lsr,covariances):
    base = importr('base')
    test = importr('tmvtnorm')
    upper = ro.IntVector([1,1])
    lower = ro.IntVector([0,0])
    x = ro.FloatVector((ws,ls))
    mapl = np.zeros((11,101))
    sum=0

    for i in range(101):
        for k in range(11):
            mean_ws = means_wsr[i,k]
            mean_ls = means_lsr[i,k]

            means = ro.FloatVector((mean_ws,mean_ls))
            t = covariances[i,k].ravel()
            v = ro.FloatVector(t)
            cov = ro.r['matrix'](v,nrow=2)
            prob = test.dtmvnorm(x,mean=means,sigma=cov,lower=lower, upper=upper)
            mapl[k,i] = prob[0]
            sum += prob[0]
    ind = np.unravel_index(np.nanargmax(mapl, axis=None), mapl.shape)
    mapl = mapl / sum
    # Marginalize across parameters; here there are 2 axis
    alpha_marginalized = np.sum(mapl,axis=0)
    beta_marginalized = np.sum(mapl,axis=1)

    new_alphas = []
    new_betas = []

    # Compute mean + standard deviation of computed distribution
    alpha_mean=0
    for i in range(101):
        alpha = i * 1/100
        alpha_running_sum = alpha*alpha_marginalized[i]
        alpha_mean = alpha_mean + alpha_running_sum
        new_alphas.append(alpha_running_sum)

    beta_mean = 0
    for k in range(11):
        beta = k
        exp_val = beta*beta_marginalized[k]
        beta_mean = beta_mean + exp_val
        new_betas.append(exp_val)

    # Get standard deviation
    std_dev_alphas = []
    for j in range (101):
        alpha = j * 1/100
        exp_val = new_alphas[j]
        std_dev_alphas.append(exp_val * (alpha - alpha_mean)**2)
    std_dev_betas=[]
    for l in range(11):
        beta = l
        exp_val = new_betas[l]
        std_dev_betas.append(exp_val * (beta - beta_mean)**2)

    alpha_stddev = np.sqrt(np.sum(std_dev_alphas))
    beta_stddev = np.sqrt(np.sum(std_dev_betas))


    #Compute Covariance
    computed_cov=0
    for j in range(101):
        for l in range(11):
            beta=l
            alpha=j*1/100
            computed_cov=computed_cov+(alpha - alpha_mean)*(beta - beta_mean)*mapl[l,j]
    computed_cov = computed_cov/(np.sqrt(np.sum(std_dev_alphas))*np.sqrt(np.sum(std_dev_betas)))


    return alpha_mean,alpha_stddev,beta_mean,beta_stddev,computed_cov

def init_model_1Back(paradigm,trial_num):
    print("Initializing 1Back model...")
    if(paradigm=="PRL"):
        if(trial_num==1000):
            print("Loading 1Back Model for 1000 Trials with PRL")
            means_wsr=np.load("means_wsr_1Back_SingleAlpha_1000Trials_PRL.npy")
            means_lsr=np.load("means_lsr_1Back_SingleAlpha_1000Trials_PRL.npy")
            covariances=np.load("covariances_1Back_SingleAlpha_1000Trials_PRL.npy") 
        else:
            print("Loading 1Back Model for 100 Trials with PRL") 
            means_wsr=np.load("means_wsr_1Back_SingleAlpha_100Trials_PRL.npy")
            means_lsr=np.load("means_lsr_1Back_SingleAlpha_100Trials_PRL.npy")
            covariances=np.load("covariances_1Back_SingleAlpha_100Trials_PRL.npy")
    else: #No PRL
        if(trial_num==1000):
            print("Loading 1Back Model for 1000 Trials without PRL")
            means_wsr=np.load("means_wsr_no_prl_1Back_SingleAlpha_1000Trials.npy")
            means_lsr=np.load("means_lsr_no_prl_1Back_SingleAlpha_1000Trials.npy")
            covariances=np.load("covariances_no_prl_1Back_SingleAlpha_1000Trials.npy")
        else:
            print("Loading 1Back Model for 100 Trials without PRL")
            means_wsr=np.load("means_wsr_no_prl_1Back_SingleAlpha.npy")
            means_lsr=np.load("means_lsr_no_prl_1Back_SingleAlpha.npy")
            covariances=np.load("covariances_no_prl_1Back_SingleAlpha.npy")
    return means_wsr,means_lsr,covariances

def getFullPosterior2Back(ws,ls,ws2,ls2,means_wsr,means_lsr,means_wsr2,means_lsr2,covariances):
    
    utils = importr('utils')
    utils.chooseCRANmirror(ind=1) 

    pkg = ['tmvtnorm']
    names_to_install = [x for **x in** pkg if not rpackages.isinstalled(x)] # the documentation has a typo, that is why I added the code here
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))
    
    base = importr('base',lib_loc='/Library/Frameworks/R.framework/Versions/3.5/Resources/library')
    test = importr('tmvtnorm',lib_loc='/Library/Frameworks/R.framework/Versions/3.5/Resources/library')
    upper = ro.IntVector([1,1,1,1])
    lower = ro.IntVector([0,0,0,0])
    x = ro.FloatVector((ws,ls,ws2,ls2))
    mapl = np.zeros((11,101))
    sum=0
    for i in range(101):
        for k in range(11):
            mean_ws = means_wsr[i,k]
            mean_ls = means_lsr[i,k]
            mean_ws2 = means_wsr2[i,k]
            mean_ls2 = means_lsr2[i,k]
            means = ro.FloatVector((mean_ws,mean_ls,mean_ws2,mean_ls2))
            t = covariances[i,k].ravel()
            v = ro.FloatVector(t)
            cov = ro.r['matrix'](v,nrow=4)
            prob = test.dtmvnorm(x,mean=means,sigma=cov,lower=lower, upper=upper)
            mapl[k,i] = prob[0]
            sum += prob[0]
    ind = np.unravel_index(np.nanargmax(mapl, axis=None), mapl.shape)
    mapl = mapl / sum

    # Marginalize across parameters; here there are 2 axis
    alpha_marginalized = np.sum(mapl,axis=0)
    beta_marginalized = np.sum(mapl,axis=1)

    new_alphas = []
    new_betas = []
    i=0
    k=0
    # Compute mean + standard deviation of computed distribution
    alpha_mean=0
    for i in range(101):
        alpha = i * 1/100
        alpha_running_sum = alpha*alpha_marginalized[i]
        alpha_mean = alpha_mean + alpha_running_sum
        new_alphas.append(alpha_running_sum)

    beta_mean = 0
    for k in range(11):
        beta = k
        exp_val = beta*beta_marginalized[k]
        beta_mean = beta_mean + exp_val
        new_betas.append(exp_val)

    # Get standard deviation
    std_dev_alphas = []
    for j in range (101):
        alpha = j * 1/100
        exp_val = new_alphas[j]
        std_dev_alphas.append(exp_val * (alpha - alpha_mean)**2)
    std_dev_betas=[]
    for l in range(11):
        beta = l
        exp_val = new_betas[l]
        std_dev_betas.append(exp_val * (beta - beta_mean)**2)

    alpha_stddev = np.sqrt(np.sum(std_dev_alphas))
    beta_stddev = np.sqrt(np.sum(std_dev_betas))


    #Compute Covariance
    computed_cov=0
    for j in range(101):
        for l in range(11):
            beta=l
            alpha=j*1/100
            computed_cov=computed_cov+(alpha - alpha_mean)*(beta - beta_mean)*mapl[l,j]
    computed_cov = computed_cov/(np.sqrt(np.sum(std_dev_alphas))*np.sqrt(np.sum(std_dev_betas)))


    return alpha_mean,alpha_stddev,beta_mean,beta_stddev,computed_cov

def init_model_2Back(paradigm,trial_num):
    print("Initializing model...")
    if(paradigm=="PRL"):
        if(trial_num==1000):
            print("Loading 2Back Model for 1000 Trials with PRL")
            means_wsr= np.load("means_wsr_2Back_SingleAlpha_1000Trials_PRL.npy")
            means_lsr= np.load("means_lsr_2Back_SingleAlpha_1000Trials_PRL.npy")
            means_wsr2= np.load("means_wsr2_2Back_SingleAlpha_1000Trials_PRL.npy")
            means_lsr2= np.load("means_lsr2_2Back_SingleAlpha_1000Trials_PRL.npy")
            covariances= np.load("covariances_2Back_SingleAlpha_1000Trials_PRL.npy")
        else:
            print("Loading 2Back Model for 100 Trials with PRL") 
            means_wsr= np.load("means_wsr_2Back_SingleAlpha_100Trials_PRL.npy")
            means_lsr= np.load("means_lsr_2Back_SingleAlpha_100Trials_PRL.npy")
            means_wsr2= np.load("means_wsr2_2Back_SingleAlpha_100Trials_PRL.npy")
            means_lsr2= np.load("means_lsr2_2Back_SingleAlpha_100Trials_PRL.npy")
            covariances= np.load("covariances_2Back_SingleAlpha_100Trials_PRL.npy")

    else: #No PRL
        if(trial_num==1000):
            print("Loading 2Back Model for 1000 Trials without PRL")
            means_wsr= np.load("means_wsr_no_prl_2Back_SingleAlpha_1000Trials.npy")
            means_lsr= np.load("means_lsr_no_prl_2Back_SingleAlpha_1000Trials.npy")
            means_wsr2= np.load("means_wsr2_no_prl_2Back_SingleAlpha_1000Trials.npy")
            means_lsr2= np.load("means_lsr2_no_prl_2Back_SingleAlpha_1000Trials.npy")
            covariances= np.load("covariances_no_prl_2Back_SingleAlpha_1000Trials.npy")
        else:
            print("Loading 2Back Model for 100 Trials without PRL")
            means_wsr= np.load("means_wsr_no_prl_2Back_100Trials.npy")
            means_lsr= np.load("means_lsr_no_prl_2Back_100Trials.npy")
            means_wsr2= np.load("means_wsr2_no_prl_2Back_100Trials.npy")
            means_lsr2= np.load("means_lsr2_no_prl_2Back_100Trials.npy")
            covariances= np.load("covariances_no_prl_2Back_100Trials.npy")

    return means_wsr,means_lsr,means_wsr2,means_lsr2,covariances


#FOR 1 BACK: input csv must have columns: subject, win_stay, lose_shift
#FOR 2 BACK: input csv must have columns: subject, win_stay, lose_shift, win_stay2, lose_shift2
df  = read_csv('filepath.csv') 
summary_df = DataFrame()

if obs_variables = "1Back":
    means_wsr,means_lsr,means_wsr2,means_lsr2,cov = init_model_2Back(paradigm,trial_num)
if obs_variables = "2Back":
    means_wsr,means_lsr,cov = init_model_1Back(paradigm,trial_num)
else: 
    print("SPECIFY PARAMETER APPROXIMATION METHOD")

for index, row in df.iterrows():
    if obs_variables = "1Back":
        print ("fitting subject", row['subject'])
        subject = (row['subject'])
        ws = (row['win_stay'])
        ls = (row['lose_shift'])
        alpha_wsls,a_st,beta_wsls,b_st,co = getFullPosterior1Back(ws,ls,means_wsr,means_lsr,cov)
    if obs_variables = "2Back":
        print ("fitting subject", row['subject'])
        subject = (row['subject'])
        ws = (row['win_stay'])
        ls = (row['lose_shift'])
        ws2 = (row['win_stay2'])
        ls2 = (row['lose_shift2'])
        alpha_wsls,a_st,beta_wsls,b_st,co = getFullPosterior2Back(ws,ls,ws2,ls2,means_wsr,means_lsr,means_wsr2,means_lsr2,cov)
    else
        print("SPECIFY PARAMETER APPROXIMATION METHOD")

    summary_df = summary_df.append( {'subj':subj_num,'
                                    'learning rate':best_simple_params[0],
                                    'learning rate std':best_simple_params[1],
                                    'inverse temperature': best_simple_ll,
                                    'inverse temperature std':best_params[0]},
                                    ignore_index = True)

summary_df.to_csv('outputfile.csv', index=False)

