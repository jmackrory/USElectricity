import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#to prevent creating huge logs.
from IPython.display import clear_output

class multiseasonal_temp(object):
    """multiseasonal_temp

    Extends multiseasonal exponential smoothing model to include temperature inputs.

    Uses analytical expressions for gradients to optimize.

    Models temperature effects via: 
    D~ A_n[T_n-T]_{+} + A_p[T-T_p]_{+}, where
    []_{+} is only non-zero if its argument is non-zero.
    """
    def __init__(self, l=0, b=0, s=np.zeros((2,24)), \
    alpha=0.1, beta=0.2, gamma=0.05*np.ones((2,2)), \
    Ap=4, An=6.5,Tp=200, Tn=150):
        self.l=l
        self.b=b
        self.s=s
        self.alpha=alpha
        self.beta=beta
        
        #easiest to initialize via a single matrix.
        self.g00 = gamma[0,0]
        self.g01 = gamma[0,1]
        self.g10 = gamma[1,0]
        self.g11 = gamma[1,1]
        #temperature model parameters.
        self.Ap = Ap
        self.An = An
        self.Tp = Tp
        self.Tn = Tn
        self.smooth_names=['alpha','beta','g00','g01','g10','g11']
        self.temp_names=['An','Ap','Tn','Tp']
        self.names=self.smooth_names+self.temp_names

        self.Ninit=4*24*7
        self.N0=24
        self.save_opt_param()

    def gamma(self):
        """gamma(self)
        Compute matrix of smoothing coefficients from instance
        variables. Allows sim to update each one algorthimically
        for numerical derivatives.
        """
        gamma=np.array([[self.g00,self.g01],[self.g10,self.g11]])
        return gamma

    def Tmodel(self,T):
        """Tmodel
        Computes temperature component of demand.
        Fits two rectified linear models (one for high temp),
        one for low.
        DT ~ Ap [T-Tp]_{+} + An[Tn-T]_{+}
        """
        m1 = T>self.Tp
        m2 = T<self.Tn
        Tm = self.Ap*( T-self.Tp)*m1 + self.An*(-T+self.Tn)*m2
        #Tm = self.Ap + self.An
        return Tm

    def fit_init_params(self,y,T):
        """fit_init_params(y)
        Fits initial parameters for Hyndman's multi-seasonal model to
        hourly electricity data. (My guess on how to do this, similar 
        to naive STL method used in statstools.timeseries)
        Finds level, bias and seasonal patterns based on first 4 weeks
        of data.  
        """
        ysub = y[:self.Ninit]
        Tsub = T[0:self.Ninit]
        ymu = np.min(ysub)
        #try to remove linear temperature trend.
        #should only try this after removing annual average?
        ysub = ysub - self.Tmodel(Tsub)
        yval = ysub.values
        ##average value
        self.l = yval[0]#np.mean(yval)
        ##average shift
        self.b = (yval[self.Ninit-1]-yval[0])/self.Ninit
        ##remove mean pattern, subtract off level, and linear trend.
        ysub = ysub-self.l-self.b*np.arange(self.Ninit)
        #mean seasonal pattern.
        #second seasonal pattern is for weekends, with days
        #Saturday/Sunday have dayofweek equal to 5 and 6.
        #make a mask to select out weekends.
        s2 = ysub.index.dayofweek >=5
        #select out weekends, and regular days. 
        y_end = ysub[s2]
        y_week= ysub[~s2]
        n1 = int(len(y_week)/self.N0)
        n2 = int(len(y_end)/self.N0)
        self.s = np.zeros((2,self.N0))
        for n in range(n1):
             self.s[0,:] = self.s[0,:]+y_week[n*self.N0:(n+1)*self.N0]/n1
        for n in range(n2):
             self.s[1,:] = self.s[1,:]+y_end[n*self.N0:(n+1)*self.N0]/n2

    def predict_dayahead(self,y,T):
        """predict_dayahead
        Predict day-ahead demand given previous parameters.
        """
        t0=y.index
        m1 = t0.dayofweek>=5
        m1_n = t0.dayofweek<5        
        m2 = t0.hour
        #find temp trend.
        Ttrend= self.Tmodel(T[t0])
        trend=self.l+self.b*np.arange(len(y))
        season=self.s[m1.astype(int),m2]
        #make prediction based on current estimates
        ypred =Ttrend+ trend+season
        return pd.Series(ypred,index=y.index)

    def correct_dayahead(self,y,ypred):
        """correct_dayahead(y,ypred)
        Updates level, bias and seasonal patterns given 
        """
        t0=y.index
        m1 = t0.dayofweek>=5
        m1_n = t0.dayofweek<5        
        eps = y-ypred
        eps_l = eps[0]#np.mean(eps)
        self.l = self.l+ self.alpha*eps_l
        eps=eps-eps_l
        eps_b = (eps[-1]-eps[0])/len(eps)
        self.b = self.b+ self.beta*eps_b
        #subtract off estimated trend
        eps=eps-eps_b*np.arange(len(eps))
        ds = np.dot(self.gamma(),np.array([m1_n,m1]))*[eps,eps]
        self.s = self.s + ds

    def predict_correct_all_days(self,y,T):
        """predict_correct_all_days
        Predict day-ahead demand given previous parameters.
        Then update parameters given true demand.
        """
        self.fit_init_params(y,T)
        t0=y.index[0]
        m1 = t0.dayofweek>=5
        m2 = t0.hour
        ypred = np.zeros(len(y))
        ti = y[:self.Ninit].index
        msk=ti.dayofweek>=5
        Ttrend= self.Tmodel(T[ti])
        trend = self.l+self.b*np.arange(self.Ninit)
        season= self.s[msk.astype(int),ti.hour.values]

        ypred[:self.Ninit] =Ttrend+trend+season
        for i in range(int(self.Ninit/self.N0),int(len(y)/self.N0)):
            tslice = slice(i*self.N0,(i+1)*self.N0)
            ypred[tslice] = self.predict_dayahead(y[tslice],T[tslice])
            self.correct_dayahead(y[tslice],ypred[tslice])
            
        ypred=pd.Series(ypred,index=y.index)        
        return ypred

    def optimize_param(self,y,T,rtol=0.01,\
                      eta=0.001,lr=0.05,nmax=1000):
        """optimize_param
        Use gradient descent to find optimum parameters for learning 
        rates alpha,beta,gamma.  Wait till all of their values are 
        settled to a relative tolerance.
        Cost is root Mean Square Error over whole time series.
        Currently tries to predict day ahead.  
        """
        self.fit_init_params(y,T)
        pred0 = self.predict_correct_all_days(y,T)
        J    = self.rmse(y[self.Ninit:],pred0[self.Ninit:])
        Ni=0
        J_opt=J
        #loop over iterations
        for i in range(nmax):
            dJ_max=0
            lr = lr*0.999
            #for each name, tweak the model's variables.
            oldJ=J            
            for n in self.names:
                #estimate finite-difference gradient.
                p0=self.__getattribute__(n)           
                self.__setattr__(n,p0*(1+eta))
                pred=self.predict_correct_all_days(y,T)
                J2=self.rmse(y[self.Ninit:],pred[self.Ninit:])
                dJ = (J2-J)/(eta*p0)
                #use mod to truncate gradient.
                if n in self.smooth_names:
                    p = p0-lr*np.fmod(dJ,1)
                else:
                    p = p0-lr*dJ
                    print(n,p,p0)
                    #restrict smoothing parameters to be within [0,1].
                p=self.check_limits(n,p,p0)
                self.__setattr__(n,p)
                J=J2
                dJ_max=max(dJ,dJ_max)
            Ni+=1       
            if (dJ_max<rtol):
                clear_output(wait=True)
                print("Hit tolerance {} at iter {}".format(dJ,Ni))
                self.plot_pred([pred,y],['Predicted','Actual'])              
                return pred
            if(Ni%5==0):
                clear_output(wait=True)                
                print("Cost, Old Cost = {},{}".format(J,oldJ))
                self.plot_pred([pred,y,pred-y],['Predicted','Actual','Error'])
            print('Iter {}.  Cost {}'.format(Ni,J))
            #pv=[]
            # for n in names:
            #     pv.append(self.__getattribute__(n))
            # pdict=dict(zip(names,pv))
            # print(pdict)
            #self.plot_pred([pred,y],['Predicted','Actual'])

            if (J<J_opt):
                J_opt=J
                self.save_opt_param()
            elif (J>1.2*J_opt):
                print('Resetting param to best')
                self.restore_opt_param()
                lr=0.8*lr
                eta=0.8*lr
                
        print("Failed to hit tolerance after {} iter\n".format(nmax))
        print("Cost:",J,J2)
        return pred 


    def calc_finite_diff_param(self,y,T,pred0,eta=0.001):
        """calc_finite_diff_param
        Compute derivatives of cost using finite difference
        """
        J  = np.sum((pred0[self.Ninit:]-y[self.Ninit:])**2)
        dparam={}
        Nt=len(y)-self.Ninit
        for n in self.temp_names:
            #estimate finite-difference gradient.
            p0=self.__getattribute__(n)           
            self.__setattr__(n,p0*(1+eta))
            pred=self.predict_correct_all_days(y,T)
            eps = pred-y;
            J2 = np.sum((eps[self.Ninit:])**2)/Nt
            dJ = (J2-J)/(eta*p0)
            dparam[n]=dJ
            self.__setattr__(n,p0)
        return dparam

    def calc_param_grad(self,y,T,pred):
        """calc_param_grad
        Uses analytical expressions for gradients to find accumulated gradient.  

        Input: y - pandas time series for demand
               T - pandas time series for temp
               pred-  pandas timeseries for prediction
        Output dparam - dict of estimated gradients of cost.
        """
        eps=pred-y
        ti=y.index
        Nt = len(eps)-self.Ninit
        #reshape so self.N0 hour periods.
        #get average for each day
        eps_level = self.calc_day_level(eps)
        eps_grad=self.calc_day_grad(eps)
        #alpha gradient
        dparam={}
        dparam['alpha']= np.sum(eps[self.Ninit+self.N0:]*eps_level[self.Ninit:-self.N0])/Nt

        dparam['beta']= np.sum(eps[self.Ninit+self.N0:]*eps_grad[self.Ninit:-self.N0])/Nt
        
        #subtract off average-trend.
        eps2 = eps-(eps_level+eps_grad)

        msk0=ti.dayofweek<5
        msk1=ti.dayofweek>=5
        
        #g_{ij} finds correction at I_i(t) due to  I_j(t-self.N0)
        dparam['g00'] = np.sum( msk0[self.Ninit+self.N0:]*msk0[self.Ninit:-self.N0]*eps2[self.Ninit:-self.N0])/Nt
        dparam['g01'] = np.sum( msk0[self.Ninit+self.N0:]*msk1[self.Ninit:-self.N0]*eps2[self.Ninit:-self.N0])/Nt
        dparam['g10'] = np.sum( msk1[self.Ninit+self.N0:]*msk0[self.Ninit:-self.N0]*eps2[self.Ninit:-self.N0])/Nt        
        dparam['g11'] = np.sum( msk1[self.Ninit+self.N0:]*msk1[self.Ninit:-self.N0]*eps2[self.Ninit:-self.N0])/Nt        

        m1 = T>self.Tp
        m2 = T<self.Tn

        m1[:self.Ninit]=False
        m2[:self.Ninit]=False        

        #initialize with zeros
        
        #Double thresholded model
        dparam['Ap'] = np.sum(eps)/Nt
        dparam['An'] = np.sum(eps)/Nt
        # dparam['Ap'] = np.sum( (T[m1]-self.Tp)*(eps[m1]))/Nt
        # dparam['An'] = np.sum( (self.Tn-T[m2])*(eps[m2]))/Nt
        dparam['Tp'] = -self.Ap*np.sum(eps[m1])/Nt
        dparam['Tn'] =  self.An*np.sum(eps[m2])/Nt
        return dparam

    def calc_day_level(self,y):
        """calc_day_level
        Given a series y, computes the average within days,
        and returns a series with the same length as y.

        Input: y - pandas time-series

        """
        Nt = len(y)
        Nd = int(Nt/self.N0)
        y2 = y.values.reshape([Nd,self.N0])
        #picks average for the day
        #y_avg=np.mean(y,axis=1,keepdims=True)
        #picks the starting value for the day.
        y_level=y2[:,0].reshape((Nd,1))
        #now repeat for each timestep
        avg_kron=np.kron(y_level,np.ones((1,self.N0))).reshape(y.shape)
        return avg_kron

    def calc_day_grad(self,y):
        """calc_day_grad
        Given a series y, computes the average gradient within days,
        and returns a series with the same length as y.

        Input: y - pandas time-series

        """
        Nt = len(y)
        Nd = int(Nt/self.N0)
        y2 = y.values.reshape([Nd,self.N0])
        y_grad=(y2[:,-1]-y2[:,0])/(self.N0-1)
        #now repeat for each timestep
        grad_kron=np.kron(y_grad,np.arange(self.N0)).reshape(y.shape)
        return grad_kron
    
    def save_opt_param(self):
        """save_opt_param

        Saves optimal parameters in variable
        """
        pv=[]
        for n in self.names:
            pv.append(self.__getattribute__(n))
        pdict=dict(zip(self.names,pv))
        self.opt_param=pdict

    def restore_opt_param(self):
        """restore_opt_param

        Restore optimal parameters in variable
        """
        for n in self.names:
            p=self.opt_param[n]
            self.__setattr__(n,p)
       
    
    def check_limits(self,n,p,p0):
        """check_limits
        Ensures smoothing and temperature parameters
        fall within reasonal ranges, e.g. [0,1] for smoothing.
        """
        if (n in ['alpha','beta','g00','g01','g10','g11']):
            if (p>1):
                print('param {} >1(!): {}'.format(n,p))
                p=min(p0+0.5*(1-p0),0.99)
            elif(p<0): 
                print('param {} <0(!): {}'.format(n,p))
                p=0.25*p0
        #restrict postive/negative temperature thresholds.        
        if (n=='Tp' and p<100):
            #nobody wants AC below 10C.
            p=100
        if (n=='Tn' and p>200):
            #nobody wants heating above 20
            p=200
        if (n in ['An','Ap'] and p<0):
            p=0.5*p0
            
        return p 

    def rmse(self,x,y):
        """compute mean_square_error"""
        z = np.sum( (x-y)*(x-y))/len(x)
        z = np.sqrt(z)
        return z
    
    def plot_pred(self,series_list,label_list):
        """make plot to compare fitted parameters"""
        for s,l in zip(series_list,label_list):
            plt.plot(s,label=l)    
        plt.legend()
        plt.show()


    def test_level(self):

        Nt = 48
        a = 4
        b = np.kron([2,3],np.ones(24))
        z=np.random.random(48)
        t = np.arange(Nt)
        y= a + np.cumsum(b);

        y_level=self.calc_day_level(y)
        y_grad=self.calc_day_grad(y)        
        
        plt.figure()
        plt.plot(y)
        plt.plot(y_level)
        plt.plot(y_grad)
        plt.show()

        if (np.sum(np.abs(y - (y_level+y_grad))) < 1E-10):
            print('Passed Grad test')
        else:
            print('Failed ya numpty')

    def test_grads(self,y,T,eta=0.01):
        
        self.fit_init_params(y,T)
        pred0 = self.predict_correct_all_days(y,T)

        dparam_ex=self.calc_param_grad(y,T,pred0)        
        dparam_fin=self.calc_finite_diff_param(y,T,pred0,eta=eta)
        for key,val in dparam_fin.items():
            print(key, val,dparam_ex[key])
            

    def plot_param_update(self,y,T,skip=960,Nupdate=3):
        """plot_param_update
        Update parameters, plot model. 
        """
        self.fit_init_params(y,T)
        t0=y.index[skip]
        m1 = t0.dayofweek>=5
        m2 = t0.hour
        ypred = np.zeros(len(y))
        ti = y[skip:skip+self.Ninit].index
        msk=ti.dayofweek>=5
        Ttrend= self.Tmodel(T[ti])
        trend = self.l+self.b*np.arange(self.Ninit)
        season= self.s[msk.astype(int),ti.hour.values]
        
        i0=skip
        i1=skip+self.Ninit
        ypred[i0:i1] =Ttrend+trend+season
        for i in range(Nupdate):
            i0=i1
            i1=i0+self.N0
            tslice = slice(i0,i1)
            ypred[tslice] = self.predict_dayahead(y[tslice],T[tslice])
            self.correct_dayahead(y[tslice],ypred[tslice])
            tarr=np.arange(i0,i1)
            ti=y[tslice].index
            msk=ti.dayofweek>=5
            Ttrend= self.Tmodel(T[ti])
            trend = self.l+self.b*np.arange(self.N0)
            season= self.s[msk.astype(int),ti.hour.values]
            plt.figure(2)
            plt.plot(ti,trend+season,'C0-')
            plt.plot(ti,y[tslice]-Ttrend,'C1-')
            plt.plot(ti,trend,'C2-')

        plt.legend(['Season','Demand-T','Trend'],
                   loc='lower right',prop={'size':9})
        plt.ylabel('Demand (kWh)')
        plt.xlabel('Time')
        plt.savefig('fig/seasonal_update.png')        
        plt.show()
        plt.figure(1)
        tarr=np.arange(skip+self.Ninit,i1)
        tslice=slice(skip+self.Ninit,i1)
        plt.plot(y.index[tslice],ypred[tslice],y.index[tslice],y[tslice])
        plt.legend(['Prediction','Actual'],
                   loc='lower right',prop={'size':9})
        plt.ylabel('Demand (kWh)')
        plt.xlabel('Time')        
        plt.savefig('fig/seasonal_err.png')
        plt.show()
        ypred=pd.Series(ypred,index=y.index)        
        return ypred
            
