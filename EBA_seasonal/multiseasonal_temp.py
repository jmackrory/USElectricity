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
    alpha=0.1, beta=0.1, gamma=np.zeros((2,2)), \
    Ap=10, An=10,Tp=200, Tn=100):
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
        return Tm

    def fit_init_params(self,y,T,ninit=4*24*7):
        """fit_init_params(y)
        Fits initial parameters for Hyndman's multi-seasonal model to
        hourly electricity data. (My guess on how to do this, similar 
        to naive STL method used in statstools.timeseries)
        Finds level, bias and seasonal patterns based on first 4 weeks
        of data.  
        """
        ysub = y[0:ninit]
        Tsub = T[0:ninit]
        ymu = np.min(ysub)
        #try to remove linear temperature trend.
        #should only try this after removing annual average?
        ysub = ysub - self.Tmodel(Tsub)
        yval = ysub.values
        ##average value
        self.l = np.mean(yval)
        ##average shift
        #self.b = (yval[ninit-1]-yval[0])/ninit
        ##remove mean pattern, subtract off level, and linear trend.
        ysub = ysub-self.l#-self.b*np.arange(ninit)
        #mean seasonal pattern.
        #second seasonal pattern is for weekends, with days
        #Saturday/Sunday have dayofweek equal to 5 and 6.
        #make a mask to select out weekends.
        s2 = ysub.index.dayofweek >=5
        #select out weekends, and regular days. 
        y_end = ysub[s2]
        y_week= ysub[~s2]
        n1 = int(len(y_week)/24)
        n2 = int(len(y_end)/24)
        self.s = np.zeros((2,24))
        for n in range(n1):
             self.s[0,:] = self.s[0,:]+y_week[n*24:(n+1)*24]/n1
        for n in range(n2):
             self.s[1,:] = self.s[1,:]+y_end[n*24:(n+1)*24]/n2

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
        trend=self.l#+self.b*np.arange(len(y))
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
        eps_l = np.mean(eps)
        self.l = self.l+ self.alpha*eps_l
        eps=eps-eps_l
        # eps_b = (eps[-1]-eps[0])/len(eps)
        # self.b = self.b+ self.beta*eps_b
        #subtract off estimated trend
        #eps=eps-eps_b*np.arange(len(eps))
        ds = np.dot(self.gamma(),np.array([m1_n,m1]))*[eps,eps]
        self.s = self.s + ds

    def STL_dayahead(self,y,T,ninit=4*24*7):
        """STL_dayahead
        Predict day-ahead demand given previous parameters.
        Then update parameters given true demand.
        """
        self.fit_init_params(y,T,ninit=ninit)
        t0=y.index[0]
        m1 = t0.dayofweek>=5
        m2 = t0.hour
        ypred = np.zeros(len(y))
        ti = y[:ninit].index
        msk=ti.dayofweek>=5
        Ttrend= self.Tmodel(T[ti])
        trend = self.l#+self.b*np.arange(ninit)
        season= self.s[msk.astype(int),ti.hour.values]

        ypred[:ninit] =Ttrend+trend+season
        for i in range(int(ninit/24),int(len(y)/24)):
            tslice = slice(i*24,(i+1)*24)
            ypred[tslice] = self.predict_dayahead(y[tslice],T[tslice])
            self.correct_dayahead(y[tslice],ypred[tslice])
            
        ypred=pd.Series(ypred,index=y.index)        
        return ypred

    def optimize_param(self,y,T,ninit=4*24*7,rtol=0.01,\
                      eta=0.001,lr=0.05,nmax=1000):
        """optimize_param
        Use gradient descent to find optimum parameters for learning 
        rates alpha,beta,gamma.  Wait till all of their values are 
        settled to a relative tolerance.
        Cost is root Mean Square Error over whole time series.
        Currently tries to predict day ahead.  
        """
        self.fit_init_params(y,T)
        #Super clunky way of specifiying names.
        #Why did I think this was superior?
        pred0 = self.STL_dayahead(y,T,ninit=ninit)
        J    = self.rmse(y[ninit:],pred0[ninit:])
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
                pred=self.STL_dayahead(y,T,ninit=ninit)
                J2=self.rmse(y[ninit:],pred[ninit:])
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

    def calc_grad(self,y,pred,T):
        """calc_grad
        Uses analytical expressions for gradients to find accumulated gradient.  
        Use fact this predict a whole 'season' at once.
        """
        
        eps=y-pred
        ti=y.index
        Nt = len(eps)
        Nday = Nt/24
        #reshape so 24 hour periods.
        #get average for each day
        eps_avg = calc_day_avg(eps)
        #alpha gradient
        dparam={}
        dparam['alpha']= np.sum(eps[t0+24:]*eps_avg[t0:-24])

        eps2 = eps-eps_avg
        eps_grad_avg=calc_day_grad(eps2)

        dparam['beta']= np.sum(eps[t0+24:]*eps_avg[t0:-24])
        
        #subtract off average-trend.
        eps2 = eps-cumsum(eps_grad_avg)


        msk0=ti.dayofweek<5
        msk1=ti.dayofweek>=5

        
        #g_{ij} finds correction at I_i(t) due to  I_j(t-24)
        dparam['g00'] = np.sum( msk0[t0+24:]*msk0[t0:-24]*eps2[t0:-24])
        dparam['g01'] = np.sum( msk0[t0+24:]*msk1[t0:-24]*eps2[t0:-24])
        dparam['g10'] = np.sum( msk1[t0+24:]*msk0[t0:-24]*eps2[t0:-24])        
        dparam['g11'] = np.sum( msk1[t0+24:]*msk1[t0:-24]*eps2[t0:-24])        

        m1 = T>self.param['Tp']
        m2 = T<self.param['Tn']
        Nt = len(T)
        #initialize with zeros
        #Double thresholded model
        dparam['Ap'] = np.sum( (T[m1]-self.Tp)*(eps[m1]))/Nt
        dparam['An'] = np.sum( (self.Tn-T[m2])*(eps[m2]))/Nt
        dparam['Tp'] = -self.Ap*np.sum(eps[m1])/Nt
        dparam['Tn'] =  self.An*np.sum(eps[m2])/Nt    
        return dparam

    def calc_day_avg(self,y):
        """calc_day_avg
        Given a series y, computes the average within days,
        and returns a series with the same length as y.

        Input: y - pandas time-series

        """
        Nt = len(y)
        Ntime=24
        Nd = int(Nt/Ntime)
        y = y.reshape([Nd,Ntime])
        #y_avg=np.mean(y,axis=1,keepdims=True)
        y_avg=y[:,0].shape
        #now repeat for each timestep
        avg_kron=np.kron(y_avg,np.ones((1,Ntime))).reshape(y.shape)
        return avg_kron

    def calc_day_grad(self,y):
        """calc_day_grad
        Given a series y, computes the average gradient within days,
        and returns a series with the same length as y.

        Input: y - pandas time-series

        """
        Nt = len(y)
        Ntime=24
        Nd = int(Nt/Ntime)
        y = y.reshape([Nd,Ntime])
        y_grad=(y[:,-1]-y[:,0])/Ntime
        #now repeat for each timestep
        grad_kron=np.kron(y_grad,np.arange(Ntime)).reshape(y.shape)
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
