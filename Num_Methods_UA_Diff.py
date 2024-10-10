import autograd.numpy as np
import numpy as npp
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.integrate import quad_vec
from autograd import jacobian
import scipy.optimize
import matplotlib.pyplot as plt

# Forward Euler Integral scheme for ODE's with large diffusion parameter

dyn_syst = "Bi-Logistic"                  # Dynamical system: "Logistic" or "Henon-Heiles"
step_h = [0.001 , 0.1]                 # Time step interval for convergence curves
step_eps = [0.001 , 1]                  # High oscillation parameter interval for the convergence curves

print(" ")
print(100*"-")
print(" Uniformly Accurate methods for 'highly diffusive' ODE's - Integral scheme")
print(100*"-")
print(" ")
print(" - Dynamical system:" , dyn_syst)
print(" - Time step inteval for convergence curves:" , step_h)
print(" - High oscillation parameter interval for the convergence curves:" , step_eps)

if dyn_syst == "Logistic":
    d = 1
    Y0 = np.array([1.5])
    A = np.array([[1]])
if dyn_syst == "Bi-Logistic":
    d = 2
    Y0 = np.array([1.5,0.5])
    A = np.array([[2,1],[1,2]])

class ODE:
    def f(tau , y , eps):
        """Vector field associated to the dynamics of the ODE.
        Inputs:
        - tau: Float - Time variable
        - y: Array of shape (d,) - Space variable
        - eps: Float - High diffusion parameter"""

        #z = (-1/eps)*A@y + y*(np.ones_like(y)-y)
        #z = (-1/eps)*A@y + np.array([y[1],y[0]])*(np.ones_like(y)-np.array([y[1],y[0]]))
        z = (-1/eps)*A@y + y*(1-y)
        return z

    def Exact_solve(T , h, eps):
        """Exact resolution of the ODE by using a very accurate Python Integrator DOP853
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter"""

        def func(t , y):
            """Function for exact solving of the ODE:
            Inputs:
            - t: Float - Time
            - y: Array of shape (1,) - Space variable"""
            return ODE.f(t , y , eps)

        S = solve_ivp(fun=func, t_span=(0, T), y0=Y0, t_eval=np.arange(0, T, h), atol=1e-13, rtol=1e-13, method="DOP853")
        Y , t = S.y , S.t
        return (Y,t)

    def Num_Solve_I(T , h , eps , print_step = True):
        """Numerical resolution of the ODE - Forward Euler Integral scheme.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - print_step: Boolean - Print the steps of computation. Default: True"""

        TT = np.arange(0, T, h)

        if print_step == True:
            print(" -> Resolution of ODE...")

        YY = np.zeros((d, np.size(TT)))
        YY[:, 0] = Y0

        for n in range(np.size(TT) - 1):
            xi, omega = npp.polynomial.legendre.leggauss(10)
            t_n = TT[n]
            t_n_1 = (TT[n] + h)
            tau = [(t_n_1-t_n)/2*ksi + (t_n+t_n_1)/2 for ksi in xi]

            def g(t,y):
                """Function used for approximation of integral in Duhamel's formulation:
                - t: Float - Time variable
                - y: Array of shape (d,) - Space variable"""
                z = ODE.f(t/eps , y , eps) + (1/eps)*A@y
                z = expm(-(t_n_1 - t)/eps*A)@z
                return z

            yy = expm(-h/eps*A)@YY[:,n]
            for i in range(len(xi)):
                yy = yy + (t_n_1-t_n)/2*omega[i]*g(tau[i],YY[:,n])
            YY[:,n+1] = yy
        return YY

    def Plot_Solve(T , h , eps , save = False):
        """Numerical resolution of the ODE vs exact solution ploted.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter
        - save: Boolean - Saves the figure or not. Default: False"""
        TT = np.arange(0,T,h)
        TT = TT.reshape(1,np.size(TT))
        Y_Exact,TT_e = ODE.Exact_solve(T,h,eps)
        Y_Num  = ODE.Num_Solve_I(T,h,eps)
        if dyn_syst == "Logistic":
            plt.figure()
            plt.scatter(TT , Y_Num , label = "Num solution" , color = "green")
            #plt.plot(np.squeeze(TT) , np.squeeze(Y_Exact) , label = "Exact solution" , color = "red")
            plt.plot(np.squeeze(TT_e) , np.squeeze(Y_Exact) , label = "Exact solution" , color = "red")
            plt.grid()
            plt.legend()
            plt.xlabel("t")
            plt.ylabel("y")
            plt.title("Exact solution vs Numerical solution - "+"$\epsilon = $"+str(eps))
            if save == True:
                plt.savefig("Integration_Micro-Macro_"+dyn_syst+"_"+num_meth+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()
        if dyn_syst == "Bi-Logistic":
            plt.figure()
            plt.scatter(TT , Y_Num[0,:] , label = "Num solution [0]" , color = "green")
            plt.scatter(TT , Y_Num[1,:] , label = "Num solution [1]" , color = "lime")
            #plt.plot(np.squeeze(TT) , np.squeeze(Y_Exact) , label = "Exact solution" , color = "red")
            plt.plot(np.squeeze(TT_e) , np.squeeze(Y_Exact[0,:]) , label = "Exact solution [0]" , color = "red")
            plt.plot(np.squeeze(TT_e) , np.squeeze(Y_Exact[1,:]) , label = "Exact solution [1]" , color = "orange")
            plt.grid()
            plt.legend()
            plt.xlabel("t")
            plt.ylabel("y")
            plt.title("Exact solution vs Numerical solution - "+"$\epsilon = $"+str(eps))
            if save == True:
                plt.savefig("Integration_Micro-Macro_"+dyn_syst+"_"+num_meth+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()
        if dyn_syst == "Henon-Heiles":

            Y_Exact_VC , Y_Num_VC = np.zeros_like(Y_Exact) , np.zeros_like(Y_Num)
            for n in range(np.size(TT)):
                VC = np.array([[np.cos(TT[0,n]/eps) , 0 , np.sin(TT[0,n]/eps) , 0] , [0 , 1 , 0 , 0] , [-np.sin(TT[0,n]/eps) , 0 , np.cos(TT[0,n]/eps) , 0] , [0 , 0 , 0 , 1]])
                Y_Exact_VC[:,n] , Y_Num_VC[:,n] = VC@Y_Exact[:,n] , VC@Y_Num[:,n]

            Ham = ODE.H(Y_Num_VC , eps)
            Ham_0 = ODE.H(Y0.reshape(d,1)@np.ones_like(TT) , eps)

            plt.figure(figsize = (12,5))
            plt.subplot(1, 2, 1)
            plt.scatter(Y_Num_VC[1, :], Y_Num_VC[3, :], s=5, label="Num solution", color="green")
            plt.plot(np.squeeze(Y_Exact_VC[1, :]), np.squeeze(Y_Exact_VC[3, :]), label="Exact solution", color="red")
            plt.grid()
            plt.legend()
            plt.xlabel("$q_2$")
            plt.ylabel("$p_2$")
            plt.title("$\epsilon = $" + str(eps))
            plt.axis("equal")
            plt.subplot(1, 2, 2)
            plt.plot(TT.reshape(np.size(TT), ), (Ham - Ham_0), label="Error on $H$")
            plt.xlabel("$t$")
            plt.ylabel("$H(y_n) - H(y_0)$")
            plt.legend()
            plt.title("Hamiltonian error")
            plt.grid()
            if save == True:
                num_meth_bis = "Forward_Euler"
                dyn_syst_bis = "Henon-Heiles"
                plt.savefig("Integration_Micro-Macro_"+dyn_syst_bis+"_"+num_meth_bis+"_T="+str(T)+"_h="+str(h)+"_epsilon="+str(eps)+".pdf")
            plt.show()
        pass


class Convergence(ODE):
    def Error(T , h , eps):
        """Computes the relative error between exact solution an numerical approximation of the solution
        w.r.t. a selected numerical method.
        Inputs:
        - T: Float - Time for ODE simulation
        - h: Float - Time step for ODE simulation
        - eps: Float - High oscillation parameter"""

        YY_exact = ODE.Exact_solve(T , h , eps)[0]
        YY_app = ODE.Num_Solve_I(T , h , eps , print_step = False)

        norm_exact = np.max(np.linalg.norm(YY_exact , 2 , axis = 0))
        norm_error = np.max(np.linalg.norm(YY_exact - YY_app, 2 , axis = 0))

        error = norm_error/norm_exact

        return error

    def Curve(T ,save = False):
        """Plots a curve of convergence w.r.t. various numerical methods
        Inputs:
        - T: Float - Time for ODE simulations
        - save: Boolean - Saves the figure or not. Default: False"""
        num_meth = "Integral_Euler"
        Num_Meths = [num_meth]
        cmap = plt.get_cmap("jet")
        Colors = [cmap(k/10) for k in range(10)]
        HH = np.exp(np.linspace(np.log(step_h[0]),np.log(step_h[1]),11))
        EPS = np.exp(np.linspace(np.log(step_eps[0]),np.log(step_eps[1]),10))
        E = np.zeros((len(Num_Meths),np.size(HH),np.size(EPS)))

        print(50 * "-")
        print("Loading...")
        print(50 * "-")
        for k in range(np.size(EPS)):
            for j in range(np.size(HH)):
                print(" - eps =  {}  \r".format(str(format(EPS[k], '.4E'))).rjust(3)," h = ",format(str(format(HH[j],'.4E'))).rjust(3), end=" ")

                for i in range(len(Num_Meths)):
                    E[i,j,k] = Convergence.Error(T , HH[j] , EPS[k])

        plt.figure()
        for k in range(np.size(EPS)):
            for i in range(len(Num_Meths)):
                plt.loglog(HH, E[i,:,k], "s" , color=Colors[k] , label = "$\epsilon = $"+str(format(EPS[k],'.2E')) , markersize = 5)
        plt.legend()
        plt.title("Integration errors - "+num_meth+" - "+dyn_syst)
        plt.xlabel("h")
        plt.ylabel("Rel. Error")
        plt.grid()
        if save == True:
            if num_meth == "Forward Euler":
                num_meth_bis = "Forward_Euler"
            if num_meth != "Forward Euler":
                num_meth_bis = num_meth
            if dyn_syst == "Henon-Heiles":
                dyn_syst_bis = "Henon-Heiles"
            if dyn_syst == "Logistic":
                dyn_syst_bis = dyn_syst
            plt.savefig("Convergence_Curve_Micro-Macro_"+dyn_syst+"_"+num_meth_bis+"_T="+str(T)+"_h="+str(step_h)+"_epsilon="+str(step_eps)+".pdf")
        plt.show()


        pass