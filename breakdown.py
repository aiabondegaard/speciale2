import numpy as np
from scipy.optimize import minimize,  NonlinearConstraint
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.") # turn of annoying warning

from EconModel import EconModelClass, jit

#alpha = 0.3

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d
from consav.linear_interp import interp_1d

class SimpleBreakdownModelClass(EconModelClass):
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

        #income
        par.alpha = 0.3 #wage premium for high education
        par.yl = 1.0 #base wage
        par.r = 0.02 #interest rate
        par.su = 0.5 #SU
        par.l = par.su #slutloan
        par.complete = 300.0 #number of ECTS for completion

        #motivation type
        par.Nm = 10 #10 different types

        par.upsilon = 0.25 #base parameter value
        par.upsilon_max = 0.4 #max base utility of studying
        par.upsilon_min = 0.3 #min base utility of studying
        par.Nupsilon = par.Nm #no. types

        par.gamma = 0.0005 #base parameter value
        par.gamma_max = 0.01 #min disutility of studying 
        par.gamma_min = 0.018 #max disutility of studying
        par.Ngamma = par.Nm

        par.tau = 5 # last period with SU
        par.t_ls = 6 #last possible period to study

        #grids
        par.Ne = 2 #Number of education types

        par.a_max = 2.0 #maximum assets
        par.a_min = -2.0 #minimum assets
        par.Na = 41 #number of grids

        par.G_max = 300.0 #max number of ECTS
        par.G_min = 0.0 #min number of ECTS
        par.Ng = 101 #number of grids 

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

        # Education grid
        par.E_grid = np.arange(par.Ne, dtype=int)

        # motivation grids grid
        par.m_grid = np.arange(par.Nm)

        par.upsilon_grid = nonlinspace(par.upsilon_min, par.upsilon_max, par.Nupsilon,1)
        par.gamma_grid = np.flip(nonlinspace(par.gamma_max, par.gamma_min, par.Ngamma,1))
        
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
        sim.e = np.zeros(shape,dtype=np.int_)
        sim.V = np.nan + np.zeros(shape)
        sim.m = np.zeros(shape,dtype=np.int_)

        #initialisation
        sim.G_init = np.zeros(par.simN)
        sim.a_init = np.zeros(par.simN)
        sim.V_init = np.zeros(par.simN)
        sim.m_init = np.random.choice(10,size=(par.simN)) #randomly choose types
        sim.e_init = np.random.choice([0.0,1.0],size=(par.simN),p=(0.4,0.6))
    
    def solve(self):
        """ solve model """
    
        #unpack
        par = self.par
        sol = self.sol

        #solve worker problem
        for t in reversed(range(par.T)):
            # print(t)
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
                        
                    else:
                        obj = lambda x: - self.value_of_choice_w(x[0],education,assets,t)

                        #bounds on consumption
                        lb_c = 0.000001 # avoid dividing with zero
                        ub_c = (self.wage_func(education)) if (assets<0.0) else (self.wage_func(education) + assets)

                        #call optimiser
                        c_init = np.array(ub_c) if i_a==0 else np.array([sol.c_w[t,i_e,i_a-1]])
                        res = minimize(obj,c_init,bounds=((lb_c,ub_c),), method='Powell')
                            
                        sol.c_w[idx_w] = res.x[0]
                        sol.V_w[idx_w] = -res.fun

        #solve
        for t in reversed(range(par.T)):
            # print(t)
            for i_m,motivation in enumerate(par.m_grid):
                for i_a,assets in enumerate(par.a_grid):
                    for i_G,credit in enumerate(par.G_grid):
                        idx = (t,i_m,i_a,i_G)

                        # value of working
                        i_e = np.int_(credit>=par.complete)
                        idx_w = (t,i_e,i_a)
                        V_work = sol.V_w[idx_w]
                        
                        if t==par.T-1: # last period
                            cons = self.cons_last(i_e, assets)

                            if cons<0.0:
                                #store results
                                sol.c_s[idx] = -1.0
                                sol.V_s[idx] = 10000000000000000000.0*cons

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

                            guess_g = ub_g if (par.complete-credit)<60 else 60.0

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



    #education status
    # def educ(self,credit):
    #     par = self.par
    #     if credit < (par.complete + 0.001):
    #         education=0.0
    #     else:
    #         education=1.0
    #     return education

    #Last period 
    def cons_last(self, education, assets):
        par = self.par

        income = par.yl*(1+par.alpha*education)
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

        income = self.wage_func(education)

        e_next = education
        #Dynamics of assets
        a_next = (1.0+par.r)*(assets + income - cons)

        #Next period utility
        V_next = sol.V_w[t+1,e_next]

        V_next_interp = interp_1d(par.a_grid,V_next,a_next)
        
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
            income = par.yl*(1+par.alpha*(credit>=par.complete))

        m_next = motivation #always unchanged
        
        #Dynamics of G
        if credit<par.complete:
            G_next = credit+g
        else:
            G_next = credit

        #dynamics of assets
        if ((credit<(par.complete-0.001)) and (g>0.001)) and ((t>par.tau) and (t<=par.t_ls)): 
            a_next = (1.0+par.r)*(assets-par.l+income-cons) #student loan in some periods
        else:
            a_next = (1.0+par.r)*(assets+income-cons)
        
        #Next period utility
        V_next = sol.V[t+1,m_next]
        V_next_interp = interp_2d(par.a_grid,par.G_grid,V_next,a_next,G_next)
        

        return (util + par.beta*V_next_interp)
    
    #disutility of studying
    def disutil_study(self,g,motivation,credit):
        par = self.par
        
        #parameter values determined by motivation type
        par.upsilon = 0.3 #par.upsilon_grid[motivation]
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
        
        return par.yl*(1+education*par.alpha)

    #income of students
    def s_wage(self,credit,t):
        par = self.par

        #if taking loans
        if (t>par.tau) and (credit<par.complete):
            return par.l
        
        #if working
        if (abs(credit-par.complete))<=0.001:
            return par.yl*(1+par.alpha*(credit>=par.complete))
        
        #while studying and not taking loans
        return par.su
    
    #Simulation
    # def simulate(self):
    #     # a. unpack
    #     par = self.par
    #     sol = self.sol
    #     sim = self.sim

    #     # b. loop over individuals and time
    #     for i in range(par.simN):

    #         # i. initialize states
    #         sim.e[i,0] = sim.e_init[i]
    #         sim.a[i,0] = sim.a_init[i]

    #         for t in range(par.simT):

    #             # ii. interpolate optimal consumption and hours
    #             idx_sol = (t,sim.e[i,t])
    #             sim.c[i,t] = interp_1d(par.a_grid,sol.c[idx_sol],sim.a[i,t])

    #             # iii. store next-period states
    #             if t<par.simT-1:
    #                 income = self.wage_func(sim.e[i,t])
    #                 sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
    #                 sim.e[i,t+1] = sim.e[i,t]
    
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

                sim.e[i,t] = (sim.G[i,t]>=par.complete)

                # store next-period states
                if t<(par.simT-1):


                    #worker income
                    income = self.wage_func(sim.e[i,t])

                    #student income
                    income_s = self.s_wage(sim.G[i,t],t)

                    #Dynamics of A
                    if  ((sim.G[i,t]<(par.complete-0.001)) and (sim.g[i,t]>0.001)) and ((t>par.tau) and (t<=par.t_ls)): 
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t]-par.l + income_s - sim.c[i,t]) #student loan in some periods

                    elif (sim.g[i,t]>0.001) and ((sim.G[i,t]<par.complete-0.001) and (t<=par.tau)):
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income_s - sim.c[i,t]) #student income in some periods
                        # print(t)
                    else:
                        sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t]) #worker income in some periods

                    #Dynamics of m7
                    sim.m[i,t+1] = sim.m[i,t]

                    #Dynamics of G
                    if (sim.G[i,t]<par.complete) and (t<=par.t_ls):
                        sim.G[i,t+1] = sim.G[i,t]+sim.g[i,t]
                    else:
                        sim.G[i,t+1] = sim.G[i,t]
                
                elif t>0:
                    if sim.g[i,t-1] <= 0.0:
                        sim.g[i,t] = 0.0

