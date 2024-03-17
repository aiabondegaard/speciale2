import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d
from consav.linear_interp import interp_1d

class SimpleSamletModelClass(EconModelClass):
    def settings(self):
        """ fundamental settings """

        pass

    def setup(self):
        #unpack
        par = self.par

        par.T=30 #periods

        #preferences
        par.beta = 0.99 #discount factor

        par.eta = -2 #CRRA coefficient

        #income
        par.alpha = 0.5 #wage premium for high education
        par.yl = 2.0 #base wage
        par.r = 0.02 #interest rate
        par.su = 0.5 #SU
        par.l = par.su #slutloan
        par.complete = 300.0 #number of ECTS for completion

        #motivation type
        par.Nm = 10 #10 different types

        par.upsilon = 0.25 #base parameter value
        par.upsilon_max = 0.05 #max base utility of studying
        par.upsilon_min = 0.001 #min base utility of studying
        par.Nupsilon = par.Nm #no. types

        par.gamma = 0.05 #base parameter value
        par.gamma_max = 0.001 #min disutility of studying 
        par.gamma_min = 0.01 #max disutility of studying
        par.Ngamma = par.Nm

        par.tau = 4 # last period with SU
        par.t_ls = 6 #last possible period to study

        #grids
        par.a_max = 2.0 #maximum assets
        par.a_min = -1 #minimum assets
        par.Na = 31 #number of grids

        par.G_max = 300.0 #max number of ECTS
        par.G_min = 0.0 #min number of ECTS
        par.Ng = 31 #number of grids 

        # simulation
        par.simT = par.T # number of periods
        par.simN = 100 # number of individuals

    def allocate(self):
        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T

        # motivation grids grid
        par.m_grid = np.arange(par.Nm)

        par.upsilon_grid = nonlinspace(par.upsilon_min, par.upsilon_max, par.Nupsilon,1)
        par.gamma_grid = np.flip(nonlinspace(par.gamma_max, par.gamma_min, par.Ngamma,1))
        
        # credit grid
        par.G_grid = nonlinspace(par.G_min,par.G_max,par.Ng,1)

        # asset grid
        par.a_grid = nonlinspace(par.a_min,par.a_max,par.Na,1)

        shape = (par.T, par.Nm, par.Na, par.Ng)

        #Solution arrays
        sol.c = np.nan + np.zeros(shape)
        sol.g = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)
        sol.V_next = np.nan + np.zeros(shape)
        sol.s = np.nan + np.zeros(shape)
        sol.succ = np.nan + np.zeros(shape)

        #simulation arrays
        shape = (par.simN, par.simT)
        sim.c = np.nan + np.zeros(shape)
        sim.g = np.nan + np.zeros(shape)
        sim.G = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.s = np.nan + np.zeros(shape)
        sim.V = np.nan + np.zeros(shape)
        sim.m = np.zeros(shape,dtype=np.int_)

        #initialisation
        sim.G_init = np.zeros(par.simN)
        sim.a_init = np.zeros(par.simN)
        sim.V_init = np.zeros(par.simN)
        sim.m_init = np.random.choice(10,size=(par.simN)) #randomly choose types
    
    def solve(self):
        """ solve model """
    
        #unpack
        par = self.par
        sol = self.sol

        #solve
        for t in reversed(range(par.T)):
            print(t)
            for i_m,motivation in enumerate(par.m_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_G,credit in enumerate(par.G_grid):
                        idx = (t,i_m,i_a,i_G)
                        #last period
                        if t==par.T-1:
                            cons = self.cons_last(assets,credit)
                            obj = self.obj_last(cons)

                            #store results
                            sol.c[idx] = cons
                            sol.g[idx] = 0.0
                            sol.V[idx] = obj
                        #after studying
                        elif t > par.t_ls:
                            obj = lambda x: - self.value_of_choice_w(x[0],motivation,credit,assets,t)

                            #bounds on consumption
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = (self.wage_func(credit)) if (assets<0.0) else (self.wage_func(credit) + assets)

                            #call optimiser
                            c_init = np.array(ub_c) if i_a==0 else np.array([sol.c[t,i_m,i_a-1,i_G]])
                            res = minimize(obj,c_init,bounds=((lb_c,ub_c),), method='Powell')

                            sol.c[idx] = res.x[0]
                            sol.g[idx] = 0.0
                            sol.V[idx] = -res.fun
                        
                        #while studying
                        else:
                            obj = lambda x: - self.value_of_choice_s(x[0],x[1],motivation,assets,credit,t)

                            # bounds on consumption 
                            lb_c = 0.000001 # avoid dividing with zero
                            ub_c = (self.s_wage(credit,t)) if (assets<0.0) else (self.s_wage(credit,t) + assets)
                            
                            # bounds on ects-points
                            lb_g = 0.0000
                            ub_g = np.minimum(80.0,par.complete-credit+1.0e-5) #avoid dividing with 0

                            bounds = ((lb_c,ub_c),(lb_g,ub_g))

                            init = np.array([ub_c,0.5*ub_g]) if i_a==0 else np.array([sol.c[t,i_m,i_a-1,i_G],sol.g[t,i_m,i_a-1,i_G]])

                            res = minimize(obj,init,bounds=bounds,method='Powell')

                            
                            sol.c[idx] = res.x[0]
                            sol.g[idx] = res.x[1]
                            sol.V[idx] = -res.fun


    #education status
    def educ(self,credit):
        par = self.par
        if credit < par.complete:
            education=0.0
        else:
            education=1.0
        return education

    #Last period 
    def cons_last(self, assets, credit):
        par = self.par

        education = self.educ(credit)
        income = par.yl*(1+par.alpha*education)
        cons = assets + income
        return cons + 1.0e-5
    
    def obj_last(self, cons):

        return self.util_w(cons) 
    
    #value function of workers
    def value_of_choice_w(self,cons,motivation,assets,credit,t):

        par = self.par
        sol = self.sol

        penalty = 0.0
        if cons < 0.0:
            penalty += cons*1_000.0
            cons = 1.0e-5

        util = self.util_w(cons)

        income = self.wage_func(credit)
        #Dynamics of G and m
        G_next = credit #unchanged while working
        m_next = motivation #always unchanged

        #Dynamics of assets
        a_next = (1.0+par.r)*(assets + income - cons)

        #Next period utility
        V_next = sol.V[t+1,m_next]
        V_next_interp = interp_2d(par.a_grid, par.G_grid,V_next,a_next,G_next)
        
        return util + par.beta*V_next_interp #+penalty
    
    #value function of students
    def value_of_choice_s(self, cons, g, motivation, assets, credit,t):
        
        par = self.par
        sol = self.sol
        
        util = self.util_s(cons,g,motivation,credit)

        #income depends on whether the student is enrolled
        if g>0.001:
            income = self.s_wage(credit,t)
        else:
            income = self.wage_func(credit)

        m_next = motivation #always unchanged
        
        #Dynamics of G
        if credit<par.complete:
            G_next = credit+g
        else:
            G_next = credit

        #dynamics of assets
        if ((credit<(par.complete-0.001)) and (g>0.001)) and ((t>=par.tau) and (t<=par.t_ls)): 
            a_next = (1.0+par.r)*(assets-par.l+income-cons) #student loan in some periods
        else:
            a_next = (1.0+par.r)*(assets+income-cons)
        
        #Next period utility
        V_next = sol.V[t+1,m_next]
        V_next_interp = interp_2d(par.a_grid,par.G_grid,V_next,a_next,G_next)
        

        return util + par.beta*V_next_interp 
    
    #disutility of studying
    def disutil_study(self,g,motivation,credit):
        par = self.par
        
        #parameter values determined by motivation type
        par.upsilon = 0.0 #par.upsilon_grid[motivation]
        par.gamma = par.gamma_grid[motivation]

        #disutility while studying
        if (credit<par.complete) and (g>0.001):
            return (par.upsilon - ((abs(par.gamma * g))**1.9))
        
        return 0.0

    #utility of students
    def util_s(self,cons,g,motivation,credit):
        par = self.par

        return ((cons**(1+par.eta))/(1+par.eta)+self.disutil_study(g,motivation,credit))
    
    #utility of workers
    def util_w(self,cons):
        par = self.par

        return cons**(1+par.eta)/(1+par.eta)
    
    #wage function of workers
    def wage_func(self,credit):
        par = self.par

        education = self.educ(credit)
        return par.yl*(1+education*par.alpha)

    #income of students
    def s_wage(self,credit,t):
        par = self.par

        #if taking loans
        if (t>=par.tau) and (credit<par.complete):
            return par.l
        
        #if working
        if (abs(credit-par.complete))<=0.001:
            return self.wage_func(credit)
        
        #while studying and not taking loans
        return par.su
    
    #Simulation
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):
            print(i)

            # i. initialize states
            sim.m[i,0] = sim.m_init[i]
            sim.a[i,0] = sim.a_init[i]
            sim.G[i,0] = sim.G_init[i]

            for i in range(par.simN):

                # initialize states
                sim.m[i,0] = sim.m_init[i]
                sim.a[i,0] = sim.a_init[i]
                sim.G[i,0] = sim.G_init[i]

                for t in range(par.simT):

                    # interpolate optimal consumption and hours
                    idx_sol = (t,sim.m[i,t])
                    sim.c[i,t] = interp_2d(par.a_grid,par.G_grid,sol.c[idx_sol],sim.a[i,t],sim.G[i,t])
                    sim.g[i,t] = interp_2d(par.a_grid,par.G_grid,sol.g[idx_sol],sim.a[i,t],sim.G[i,t])

                    # store next-period states
                    if t<(par.simT-1):
                        #worker income
                        income = self.wage_func(sim.G[i,t])

                        #student income
                        income_s = self.s_wage(sim.G[i,t],t)

                        #Dynamics of A
                        if  ((sim.G[i,t]<(par.complete-0.001)) and (sim.g[i,t]>0.001)) and ((t>=par.tau) and (t<par.t_ls)): 
                            sim.a[i,t+1] = (1+par.r)*(sim.a[i,t]-par.l + income_s - sim.c[i,t]) #student loan in some periods

                        elif ((sim.G[i,t]<par.complete) and (t<=par.tau)) and (sim.g[i,t]>0.001):
                            sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income_s - sim.c[i,t]) #student income in some periods

                        else:
                            sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t]) #worker income in some periods

                        #Dynamics of m
                        sim.m[i,t+1] = sim.m[i,t]

                        #Dynamics of G
                        if (sim.G[i,t]<par.complete) and (t<=par.t_ls):
                            sim.G[i,t+1] = sim.G[i,t]+sim.g[i,t]
                        else:
                            sim.G[i,t+1] = sim.G[i,t]

