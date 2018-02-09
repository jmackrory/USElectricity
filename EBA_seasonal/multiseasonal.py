"""
Class for Multiseasonal demand model, ignoring temperature.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Class for Multiseasonal model.
#Implements multi-seasonal smoothing with two seasons.
class multiseasonal(object):
    """
    Create multiseasonal demand model, ignoring temperature.
    Key parameters are:
    l - average level 
    b - average gradient
    s - seasonal pattern (2x24), with daily/weekend variants.
    These are updated with parameters
    alpha - l update
    beta  - b update
    gamma - 2x2 matrix for s update.
    """
    def __init__(self, l=0,b=0,s=np.zeros((2,24)),  alpha=0.1,beta=0.1,gamma=0.1*np.ones((2,2))):
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

    def gamma(self):
        gamma=np.array([[self.g00,self.g01],[self.g10,self.g11]])
        return gamma

    def fit_init_params(self,y,ninit=4*24*7):
        """fit_init_params(y)
        Fits initial parameters for Hyndman's multi-seasonal model to
        hourly electricity data.
        (My guess on how to do this, similar to naive STL method used in 
        statstools.timeseries)
        Finds level, bias and seasonal patterns based on first 4 weeks of data.  
        """
        ysub = y[0:ninit]
        yval = ysub.values
        ##average value
        self.l = np.mean(yval)
        ##average shift
        self.b = (yval[ninit-1]-yval[0])/ninit
        ##remove mean pattern, subtract off level, and linear trend.
        ysub = ysub-self.l-self.b*np.arange(ninit)
        #Compute mean seasonal patterns.
        #FIrst pattern is for weekdays, and second pattern is for weekends.
        #with days Saturday/Sunday have dayofweek equal to 5 and 6.
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

    def predict_correct_onestep(self,yhat,ypred,t):
        """predict_correct_onestep
        Updates time parameters based on differences between predicted 
        and measured.  Also predicts next step based on current values. 
        (Not fixing the model parameters for update strength!)
        Work in Progress
        """
        eps=(yhat-ypred)
        #find seasonal patterns.  
        m1 = t.hour                     
        mr  = int(t.dayofweek>=5)
        nmr = int(t.dayofweek<5)         
        #Original version with bias in there too?
        self.l = self.l + self.b + self.alpha*eps
        self.b = self.b + self.beta*eps
        #ynew = self.l + self.s[mr,m1]         
        # self.l = self.l +self.alpha*eps         
        # self.b = self.b + self.beta*eps
        # ynew = self.l  + self.s[mr,m1]
        
        #update row, and hour
        ds = np.dot(self.gamma(),np.array([nmr,mr]))*eps
        self.s[:,m1] = self.s[:,m1]+ds

        #predict the next value given the updated parameters
        ynew = self.l + self.s[mr,m1]
        return ynew

    def STL_onestep(self,y,ninit=4*24*7):
        """STL_onestep
        Generates initial parameters, and then predicts remainder
        of series for input data y.  
        """
        self.fit_init_params(y,ninit=ninit)
        ypred = np.zeros(len(y))         
        t0=y.index[0]
        m1 = t0.dayofweek>=5
        m2 = t0.hour
        
        ti = y[:ninit].index
        msk=ti.dayofweek>=5
        ypred[:ninit] = self.l+self.b*np.arange(ninit) \
                 + self.s[msk.astype(int),ti.hour.values]
        for i in range(ninit,len(y)):
            ynew=self.predict_correct_onestep(y[i],
                    ypred[i-1],y.index[i])
            ypred[i] = ynew
        # if i%(24*7) ==0:
        #         print("l: {} b: {}\n".format(str(l),str(b)))
        #         print(s,"\n")
        ypred=pd.Series(ypred,index=y.index)         
        return ypred

    def predict_correct_dayahead(self,y):
        """predict_correct_dayahead
        Predict day-ahead demand given previous parameters.
        Then update parameters given true demand.
        """
        t0=y.index
        m1 = t0.dayofweek>=5
        m1_n = t0.dayofweek<5         
        m2 = t0.hour
        trend=self.l+self.b*np.arange(len(y))
        season=self.s[m1.astype(int),m2]
        ypred = trend+season
        #parameter updates
        #Is this right?  Seems wrong to me.  
        eps = y-ypred
        eps_l = np.mean(eps)
        self.l = self.l + self.alpha*eps_l
        eps=eps-eps_l
        eps_b = (eps[-1]-eps[0])/len(eps)
        self.b = self.b + self.beta*eps_b
        eps=eps-eps_b*np.arange(len(eps))
        ds = np.dot(self.gamma(),np.array([m1_n,m1]))*[eps,eps]
        self.s = self.s + ds
        ytot=pd.Series(ypred,index=y.index)
        return ytot

    def STL_dayahead(self,y,ninit=4*24*7):
        """STL_dayahead
        Predict day-ahead demand given previous parameters.
        Then update parameters given true demand.
        """
        self.fit_init_params(y,ninit=ninit)
        t0=y.index[0]
        m1 = t0.dayofweek>=5
        m2 = t0.hour
        ypred = np.zeros(len(y))
        ti = y[:ninit].index
        msk=ti.dayofweek>=5
        ypred[:ninit] = self.l+self.b*np.arange(ninit) \
                 + self.s[msk.astype(int),ti.hour.values]
                 
        for i in range(int(ninit/24),int(len(y)/24)):
            tslice = slice(i*24,(i+1)*24)
            ypred[tslice] = self.predict_correct_dayahead(y[tslice])
            
        # if i%(24*7) ==0:
        #         print("l: {} b: {}\n".format(str(l),str(b)))
        #         print(s,"\n")
        ypred=pd.Series(ypred,index=y.index)         
        return ypred

    def optimize_param(self,y,ninit=4*24*7,rtol=0.01,\
                        eta=0.01,lr=0.05,nmax=100):
        """optimize_param
        Use gradient descent to find optimum parameters for learning 
        rates alpha,beta,gamma.  Wait till all of their values are 
        settled to a relative tolerance.
        Cost is root Mean Square Error over whole time series.
        Currently tries to predict day ahead.  
        Input:
        y - series to fit
        ninit - number of values to use in initial parameter fitting
        rtol - relative tolerance on parameters
        eta - fraction for finite-difference step
        lr  - learning rate
        nmax - maximum number of iterations.
        """
        self.fit_init_params(y)
        #Super clunky way of specifiying names.
        #Why did I think this was superior?
        names=['alpha','beta','g00','g01','g10','g11']
        pred0 = self.STL_dayahead(y,ninit=ninit)
        J    = self.rmse(y[ninit:],pred0[ninit:])
        Ni=0
        oldJ = J
        #loop over iterations
        for i in range(nmax):
            dJ_max=0
            #for each name, tweak the model's variables.
            eta=eta*0.99
            for n in names:
                #do finite-difference estimate of update.
                p0=self.__getattribute__(n)            
                self.__setattr__(n,p0*(1+eta))
                pred=self.STL_dayahead(y,ninit=ninit)
                J2=self.rmse(y[ninit:],pred[ninit:])
                dJ = np.abs((J2-J)/J)
                # if (debug):
                #     print('J,J2,p',J,J2,p0)
                #actually update 
                #p = p0-lr*(J2-J)/(eta*p0)
                #crop gradient to within +/- 1
                p = p0+lr*np.fmod(dJ/(eta*p0),1)
                if (p>1):
                    print('param {} >1(!): {}'.format(n,p))
                    print('J,J2,p',J,J2,p0)
                    p=min(p0+0.5*(1-p0),0.99)
                elif(p<0): 
                    print('param {} <0(!): {}'.format(n,p))
                    print('J,J2,p',J,J2,p0)
                    p=0.5*p0
                self.__setattr__(n,p)
                J=J2
                dJ_max=max(dJ,dJ_max)
            Ni+=1       
            if (dJ_max<rtol):
                print("Hit tolerance {} at iter {}".format(dJ,Ni))
                self.plot_pred([pred,y],['Predicted','Actual'])               
                return pred
            if(Ni%10==0):
                print("Cost, Old Cost = {},{}".format(J,oldJ))
                self.plot_pred([pred,y],['Predicted','Actual'])
                for n in names:
                    p0=self.__getattribute__(n)
                    print(n,p0)
                OldJ=J    

        print("Failed to hit tolerance after {} iter\n".format(iter))
        print("Cost:",J,J2)
        return pred 

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

    
#End of Class
