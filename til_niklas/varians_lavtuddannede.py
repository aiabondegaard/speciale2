import numpy as np
import math
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass, jit

#alpha = 0.3

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d
from consav.linear_interp import interp_1d
from consav.quadrature import log_normal_gauss_hermite

class VariansLavtuddClass(EconModelClass):
    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        #unpack
        par = self.par

        par.T=49 #periods

        #preferences
        par.beta = 0.98 #discount factor

        par.eta = -2.0 #CRRA coefficient

        par.rho = 1.3 #disutility of studying

        par.sigma_s = 0.3 #standard deviation
        par.sigma_w_l = 0.15 #standard deviation of wage for low education
        par.sigma_w_b = 0.1 #standard deviation of wage for bachelor education
        par.sigma_w_m = 0.05 #standard deviation of wage for master education
        
        #par.scale = par.sigma*np.sqrt(6)/math.pi #GEV scale
        #par.location = -(par.scale*np.euler_gamma) #generate mean zero shocks

        #income
        par.alpha = 0.15 #wage premium for bachelor education
        par.yl = 1.0 #base wage
        par.r = 0.02 #interest rate
        par.su = 0.16 #SU
        par.s_w = 0.14 #wage while studying
        par.l = par.su #slutloan
        par.complete = 300.0 #number of ECTS for masters degree
        par.bachelor = 180.0 #number of ECTS for bachelors degree

        par.tax = 0.4 #tax rate
        par.tax_s = 0.0 #tax rate

        #motivation type
        par.Nm = 5 #5 different types

        par.upsilon = 0.8 #base parameter value
        par.upsilon_max = 0.4 #max base utility of studying
        par.upsilon_min = 0.3 #min base utility of studying
        par.Nupsilon = par.Nm #no. types

        par.gamma = 0.0005 #base parameter value
        par.gamma_max = 0.01 #min disutility of studying
        par.gamma_min = 0.018 #max disutility of studying
        par.Ngamma = par.Nm

        par.tau = 4 # last period with SU
        par.t_ls = 6 #last possible period to study
        par.t_ls_b = 3 #last possible period to study bachelor

        #grids
        par.Ne = 3 #Number of education types

        par.a_max = 2.0 #maximum assets
        par.a_min = -2.0 #minimum assets
        par.Na = 41 #number of grids

        par.G_max = 300.0 #max number of ECTS
        par.G_min = 0.0 #min number of ECTS
        par.Ng = 101 #number of grids 

        par.Nxi = 10 #number of quadrature points
        par.Npsi = 10 #number of quadrature points

        # seed
        par.seed = 69

        # simulation
        par.simT = par.T # number of periods
        par.simN = 50000 # number of individuals

    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # Education grid
        par.E_grid = np.arange(par.Ne, dtype=int)

        # motivation grids grid
        par.m_grid = np.arange(par.Nm)

        par.upsilon_grid = nonlinspace(par.upsilon_min, par.upsilon_max, par.Nupsilon,1)
        par.gamma_grid = np.flip(nonlinspace(par.gamma_max, par.gamma_min, par.Ngamma,1))

        # shocks
        par.xi_grid,par.xi_weight = log_normal_gauss_hermite(par.sigma_s,par.Nxi,mu=1.0)

        #no positive ects-shocks
        for i in range(par.Nxi):
            if par.xi_grid[i]>1.0:
                    par.xi_grid[i] = 1.0

        # income shocks
        par.psi_grid_l,par.psi_weight_l = log_normal_gauss_hermite(par.sigma_w_l,par.Npsi,mu=1.0)
        par.psi_grid_b,par.psi_weight_b = log_normal_gauss_hermite(par.sigma_w_b,par.Npsi,mu=1.0)
        par.psi_grid_m,par.psi_weight_m = log_normal_gauss_hermite(par.sigma_w_m,par.Npsi,mu=1.0)
        
        # credit grid
        par.G_grid = nonlinspace(par.G_min,par.G_max,par.Ng,1)

        # asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1)

        shape_s = (par.T, par.Nm, par.Na, par.Ng)
        shape_w = (par.T, par.Ne, par.Na)

        #Student solution arrays
        sol.c_s = np.nan + np.zeros(shape_s)
        sol.g = np.nan + np.zeros(shape_s)
        sol.V_s = np.nan + np.zeros(shape_s)
        sol.V = np.nan + np.zeros(shape_s)
        sol.c = np.nan + np.zeros(shape_s)


        #Worker solution arrays
        sol.c_w = np.nan + np.zeros(shape_w)
        sol.V_w = np.nan + np.zeros(shape_w)


        #simulation arrays
        shape = (par.simN, par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.g = np.nan + np.zeros(shape)
        sim.G = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.e = np.nan + np.zeros(shape)
        sim.U = np.nan + np.zeros(shape)
        sim.V = np.nan + np.zeros(shape)
        sim.m = np.zeros(shape,dtype=np.int_)
        sim.income = np.nan + np.zeros(shape)
        sim.su = np.nan + np.zeros(shape)
        
        #shock simulation
        rng = np.random.default_rng(seed=par.seed) #set seed
        #sim.xi = rng.gumbel(loc=0.0, scale=par.scale, size=shape)

        #ECTS-shocks
        sim.xi = np.exp(par.sigma_s*rng.normal(size=shape) - 0.5*par.sigma_s**2)

        #no positive ects-shocks
        for i in range(par.simN):
            for t in range(par.simT):
                if sim.xi[i,t]>1.0:
                    sim.xi[i,t] = 1.0
        
        #income shocks
        sim.psi_l = np.exp(par.sigma_w_l*rng.normal(size=shape) - 0.5*par.sigma_w_l**2)
        sim.psi_b = np.exp(par.sigma_w_b*rng.normal(size=shape) - 0.5*par.sigma_w_b**2)
        sim.psi_m = np.exp(par.sigma_w_m*rng.normal(size=shape) - 0.5*par.sigma_w_m**2)

        #initialisation
        sim.G_init = np.zeros(par.simN)
        sim.a_init = np.zeros(par.simN)
        sim.V_init = np.zeros(par.simN)
        sim.m_init = np.random.choice(5,size=(par.simN)) #randomly choose types
        sim.e_init = np.random.choice([0.0,1.0,2.0],size=(par.simN),p=(0.2,0.2,0.6))
    
    def solve(self):
        """ solve model """
    
        #unpack
        par = self.par
        sol = self.sol

        #solve worker problem
        for t in reversed(range(par.T)):
            print(t)
            for i_e, education in enumerate(par.E_grid):
                for i_a,assets in enumerate(par.a_grid):
                    idx_w = (t,i_e,i_a)

                    if t==par.T-1: # last period
                        cons = self.cons_last(education, assets)

                        if cons<0.0:
                            #store results
                            sol.c_w[idx_w] = -1.0
                            sol.V_w[idx_w] = 10000000000000000000.0*cons
                        
                        else:

                            obj = self.obj_last(cons)
                            #store results
                            sol.c_w[idx_w] = cons
                            sol.V_w[idx_w] = obj
                            sol.g[idx_w] = 0.0

                    elif t>(par.t_ls+15):
                        if assets<0.0:
                            sol.c_w[idx_w] = 0.000001
                            sol.V_w[idx_w] = 1000000.0*assets
                        
                        else:
                            obj = lambda x: - self.value_of_choice_w(x[0],education,assets,t)

                            #bounds on consumption
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = (self.wage_func(education)*(1-par.tax)) if (assets<0.0) else (self.wage_func(education)*(1-par.tax) + assets)

                            #call optimiser
                            c_init = np.array(ub_c) if i_a==0 else (np.array([sol.c_w[t,i_e,i_a-1]]) if np.array([sol.c_w[t,i_e,i_a-1]])>0 else np.array([0.000001]))
                            res = minimize(obj,c_init,bounds=((lb_c,np.inf),), method='Nelder-Mead')

                            sol.c_w[idx_w] = res.x[0]
                            sol.V_w[idx_w] = -res.fun


                    else:
                        obj = lambda x: - self.value_of_choice_w(x[0],education,assets,t)

                        #bounds on consumption
                        lb_c = 0.000001 # avoid dividing with zero
                        ub_c = (self.wage_func(education)*(1-par.tax)) if (assets<0.0) else (self.wage_func(education)*(1-par.tax) + assets)

                        #call optimiser
                        c_init = np.array(ub_c) if i_a==0 else np.array([sol.c_w[t,i_e,i_a-1]])
                        res = minimize(obj,c_init,bounds=((lb_c,np.inf),), method='Nelder-Mead')
                            
                        sol.c_w[idx_w] = res.x[0]
                        sol.V_w[idx_w] = -res.fun

        #solve
        for t in reversed(range(par.T)):
            print(t)
            for i_m,motivation in enumerate(par.m_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_G,credit in enumerate(par.G_grid):
                        idx = (t,i_m,i_a,i_G)

                        # value of working
                        i_e = self.educ_level(credit)
                        idx_w = (t,i_e,i_a)
                        V_work = sol.V_w[idx_w]
                        
                        if t==par.T-1: # last period
                            cons = self.cons_last(i_e, assets)

                            if cons<0.0:
                                #store results
                                sol.c_s[idx] = -1.0
                                sol.V_s[idx] = 10000000000000000000.0*cons
                            
                            else:
                                obj = self.obj_last(cons)
                                #store results
                                sol.g[idx] = 0.0
                                sol.c_s[idx] = cons
                                sol.V_s[idx] = obj

                        elif t>par.t_ls:
                            sol.V_s[idx] = V_work
                            sol.c_s[idx] = sol.c_w[idx_w]
                            sol.g[idx] = 0.0
                        
                        else:
                            obj = lambda x: - self.value_of_choice_s(x[0],x[1],motivation,assets,credit,t)

                            #bounds on consumption
                            lb_c = 0.000001
                            ub_c = (self.s_wage(credit,t)) if (assets<0.0) else (self.s_wage(credit,t) + assets)

                            # bounds on ects-points
                            lb_g = 0.0000
                            ub_g = (par.complete-credit+1.0e-5) if (credit<par.complete+1.0e-5) else 1.0e-5
                            #ub_g = np.minimum(300,par.complete-credit+1.0e-5) #avoid dividing with 0 

                            guess_g = (par.complete-credit) if (par.complete-credit)<=60 else 60.0

                            bounds = ((lb_c,ub_c),(lb_g,ub_g))

                            init = np.array([ub_c,guess_g]) if i_a==0 else np.array([sol.c_s[t,i_m,i_a-1,i_G],sol.g[t,i_m,i_a-1,i_G]])#else np.array([sol.c_s[t,i_m,i_a-1,i_G],sol.g[t,i_m,i_a-1,i_G]])

                            res = minimize(obj,init,bounds=bounds,method='Nelder-Mead')

                            sol.c_s[idx] = res.x[0]
                            sol.g[idx] = res.x[1]
                            sol.V_s[idx] = -res.fun
                            
                        #max utility
                        sol.V[idx] = np.max([sol.V_s[idx],V_work])

                        if sol.V[idx] == sol.V_s[idx]:
                            sol.c[idx] = sol.c_s[idx]
                        else:
                            sol.c[idx] = sol.c_w[idx_w]
                            sol.g[idx] = 0.0

                        # if sol.V[idx] == V_work:
                        #     sol.c[idx] = sol.c_w[idx_w]
                        #     sol.g[idx] = 0.0
                        # else:
                        #     sol.c[idx] = sol.c_s[idx]
                            



    #education status
    # def educ(self,credit):
    #     par = self.par
    #     if credit < (par.complete + 0.001):
    #         education=0.0
    #     else:
    #         education=1.0
    #     return education

    def educ_level(self, credit):
        par = self.par

        if credit < par.bachelor-1.0e-5:
            return 0  # Uddannelsesniveau 0
        elif credit < par.complete-1.0e-5:  # Antager at par.complete er 300 for fuld uddannelse
            return 1  # Uddannelsesniveau 1
        else:
            return 2  # Uddannelsesniveau 2 (fuld uddannelse)
        
    #Last period 
    def cons_last(self, education, assets):
        par = self.par

        income = self.wage_func(education)*(1-par.tax)
        cons = assets + income
        return cons + 1.0e-5
    
    def obj_last(self, cons):

        return self.util_w(cons) 
    
    #value function of workers
    def value_of_choice_w(self,cons,education,assets,t):

        par = self.par
        sol = self.sol

        # penalty = 0.0
        # if cons < 0.0:
        #     penalty += cons*1_000.0
        #     cons = 1.0e-5

        util = self.util_w(cons)

        income = self.wage_func(education)*(1-par.tax)

        e_next = education
        
        # loop over ects shocks
        EV_next = 0.0
        if education==0:
            for i_psi,psi in enumerate(par.psi_grid_l):
                #Dynamics of assets
                a_next = (1.0+par.r)*(assets + income*par.psi_grid_l[i_psi] - cons)

                #Next period utility
                V_next = sol.V_w[t+1,e_next]
                V_next_interp = interp_1d(par.a_grid,V_next,a_next)

                EV_next += V_next_interp*par.psi_weight_l[i_psi]

        elif education==1:
            for i_psi,psi in enumerate(par.psi_grid_b):
                #Dynamics of assets
                a_next = (1.0+par.r)*(assets + income*par.psi_grid_b[i_psi] - cons)

                #Next period utility
                V_next = sol.V_w[t+1,e_next]
                V_next_interp = interp_1d(par.a_grid,V_next,a_next)

                EV_next += V_next_interp*par.psi_weight_b[i_psi]
        
        elif education==2:
            for i_psi,psi in enumerate(par.psi_grid_m):
                #Dynamics of assets
                a_next = (1.0+par.r)*(assets + income*par.psi_grid_m[i_psi] - cons)

                #Next period utility
                V_next = sol.V_w[t+1,e_next]
                V_next_interp = interp_1d(par.a_grid,V_next,a_next)

                EV_next += V_next_interp*par.psi_weight_m[i_psi]
        
        else:
            print("Error: Education level not defined")
        
        return util + par.beta*EV_next #+penalty
    
    #value function of students
    def value_of_choice_s(self, cons, g, motivation, assets, credit,t):
        
        par = self.par
        sol = self.sol
        
        util = self.util_s(cons,g,motivation,credit)

        #income depends on whether the student is enrolled
        if g>0.001:
            income = self.s_wage(credit,t)
        else:
            income = 0.000001
            #income = par.yl*(1+par.alpha*(self.educ_level(credit)))

        m_next = motivation #always unchanged
        
        #dynamics of assets
        if ((credit<(par.complete)) and (g>0.001)) and ((t>par.tau) and (t<=par.t_ls)): 
            a_next = (1.0+par.r)*(assets-par.l+income-cons) #student loan in some periods
        else:
            a_next = (1.0+par.r)*(assets+income-cons)

        # loop over ects shocks
        EV_next = 0.0
        for i_xi,xi in enumerate(par.xi_grid):
            #Dynamics of G
            if credit<par.bachelor and t>par.t_ls_b: #expelled if bachelor not finished in 4 years
                G_next = credit
            elif (par.complete>credit>(par.complete-60)) or (t==par.t_ls):
                G_next = credit+g
            elif (par.bachelor>credit>(par.bachelor-60)) or (t==par.t_ls_b):
                G_next = credit+g
            elif credit<par.complete:
                G_next = credit+g*par.xi_grid[i_xi]
            else:
                G_next = credit
        
            #Next period utility
            V_next = sol.V[t+1,m_next]
            V_next_interp = interp_2d(par.a_grid,par.G_grid,V_next,a_next,G_next)

            # weight the interpolated value with the likelihood
            EV_next += V_next_interp*par.xi_weight[i_xi]


        return (util + par.beta*EV_next)
    
    #disutility of studying
    def disutil_study(self,g,motivation,credit):
        par = self.par
        
        #parameter values determined by motivation type
        #par.upsilon = 20.0 #par.upsilon_grid[motivation]
        par.gamma = par.gamma_grid[motivation]

        #disutility while studying
        if (credit<par.complete) and (g>0.001):
            return (par.upsilon - ((abs(par.gamma * g))**(par.rho)))
        
        return 0.0

    #utility of students
    def util_s(self,cons,g,motivation,credit):
        par = self.par

        return (((cons**(1+par.eta))/(1+par.eta))+(self.disutil_study(g,motivation,credit)))
    
    #utility of workers
    def util_w(self,cons):
        par = self.par

        return cons**(1+par.eta)/(1+par.eta)
    
    #wage function of workers
    def wage_func(self,education):
        par = self.par
        
        return par.yl*(1+education*par.alpha) #pre tax income

    #income of students
    def s_wage(self,credit,t):
        par = self.par


        #if taking loans
        if (t>par.tau) and (credit<par.complete):
            return par.l + par.s_w*(1-par.tax_s)
        
        #if working
        # if (abs(credit-par.complete))<=0.001:
        #     return par.yl*(1+par.alpha*(credit>=par.complete))
        
        #while studying and not taking loans
        return par.su + par.s_w*(1-par.tax_s)

    
    #Simulation
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.m[i,0] = sim.m_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.G[i,0] = sim.G_init[i]
            sim.e[i,0] = np.int_(sim.G[i,0]>=par.complete)

            for t in range(par.simT):

                # interpolate optimal consumption and hours
                idx_sol = (t,sim.m[i,t])
                sim.c[i,t] = interp_2d(par.a_grid,par.G_grid,sol.c[idx_sol],sim.a[i,t],sim.G[i,t])
                sim.g[i,t] = interp_2d(par.a_grid,par.G_grid,sol.g[idx_sol],sim.a[i,t],sim.G[i,t])

                if sim.G[i,t]>=par.complete-1.0e-5:
                    sim.e[i,t] = 2
                elif sim.G[i,t]>=(par.bachelor-1.0e-5) and sim.g[i,t]<=1.0e-5:
                    sim.e[i,t] = 1
                else:
                    sim.e[i,t] = 0
                
                #utility
                sim.U[i,t] = self.util_s(sim.c[i,t],sim.g[i,t],sim.m[i,t],sim.G[i,t])

                #discounted utility
                sim.V[i,t] = sim.U[i,t]*(par.beta**t)

                # store next-period states
                if t<(par.simT-1):


                    #worker income
                    income = self.wage_func(sim.e[i,t])*(1-par.tax)

                    #student income
                    income_s = self.s_wage(sim.G[i,t],t)

                    #Dynamics of A
                    if  ((sim.G[i,t]<(par.complete)) and (sim.g[i,t]>0.001)) and ((t>par.tau) and (t<=par.t_ls)): 
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t]-par.l + income_s - sim.c[i,t]) #student loan in some periods
                        
                        sim.income[i,t] = 0.0 #no income while studying

                        sim.su[i,t] = 0.0 #no SU

                    elif (sim.g[i,t]>0.001) and ((sim.G[i,t]<par.complete) and (t<=par.tau)):
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income_s - sim.c[i,t]) #student income in some periods
                        
                        sim.income[i,t] = 0.0 #no income while studying

                        sim.su[i,t] = par.su #SU

                    else:
                        if sim.e[i,t]==0:
                            sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income*sim.psi_l[i,t] - sim.c[i,t]) #worker income in some periods

                            sim.income[i,t] = self.wage_func(sim.e[i,t])*sim.psi_l[i,t] #worker income
                        
                        elif sim.e[i,t]==1:
                            sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income*sim.psi_b[i,t] - sim.c[i,t])

                            sim.income[i,t] = self.wage_func(sim.e[i,t])*sim.psi_b[i,t] #worker income
                        
                        elif sim.e[i,t]==2:
                            sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income*sim.psi_m[i,t] - sim.c[i,t])

                            sim.income[i,t] = self.wage_func(sim.e[i,t])*sim.psi_m[i,t]
                        
                        sim.su[i,t] = 0.0 #no SU

                    #Dynamics of m
                    sim.m[i,t+1] = sim.m[i,t]

                    #Dynamics of G
                    if ((par.complete)>sim.G[i,t]>(par.complete-60)) or (t==par.t_ls):
                        sim.G[i,t+1] = sim.G[i,t]+sim.g[i,t]
                    elif ((par.bachelor)>sim.G[i,t]>(par.bachelor-60)) or (t==par.t_ls_b):
                        sim.G[i,t+1] = sim.G[i,t]+sim.g[i,t]
                    elif (sim.G[i,t]<(par.complete-60)) and (t<par.t_ls):
                        sim.G[i,t+1] = sim.G[i,t]+sim.g[i,t]*sim.xi[i,t]
                    else:
                        sim.G[i,t+1] = sim.G[i,t]

                    if t>0:
                        if sim.g[i,t-1] <= 0.0:
                            sim.g[i,t] = 0.0

