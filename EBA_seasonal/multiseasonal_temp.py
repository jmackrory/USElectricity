class multiseasonal_temp(object):
    def __init__(self, l=0, b=0, s=np.zeros((2,24)), \
    alpha=0.1, beta=0.1, gamma=np.zeros((2,2)), \
    A0=2000, Ap=10, An=10,Tp=200, Tn=100):
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

    def gamma(self):
        """gamma(self)
        Compute matrix of smoothing coefficients from instance
        variables. Allows sim to update each one algorthimically
        for numerical derivatives.
        """
        gamma=np.array([[self.g00,self.g01],[self.g10,self.g11]])
        return gamma

    def Tmodel(self,T):
        m1 = T>self.Tp
        m2 = T<self.Tn
        Tm = self.Ap*( T-self.Tp)*m1 \
                  +self.An*(-T+self.Tn)*m2
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
        self.b = (yval[ninit-1]-yval[0])/ninit
        ##remove mean pattern, subtract off level, and linear trend.
        ysub = ysub-self.l-self.b*np.arange(ninit)
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

    def predict_correct_dayahead(self,y,T):
        """predict_correct_dayahead
        Predict day-ahead demand given previous parameters.
        Then update parameters given true demand.
        """
        t0=y.index
        m1 = t0.dayofweek>=5
        m1_n = t0.dayofweek<5        
        m2 = t0.hour
        Ttrend= self.Tmodel(T[t0])
        trend=self.l+self.b*np.arange(len(y))
        season=self.s[m1.astype(int),m2]
        ypred =Ttrend+ trend+season

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
        trend = self.l+self.b*np.arange(ninit)
        season= self.s[msk.astype(int),ti.hour.values]

        ypred[:ninit] =Ttrend+trend+season
        for i in range(int(ninit/24),int(len(y)/24)):
            tslice = slice(i*24,(i+1)*24)
            ypred[tslice] = self.predict_correct_dayahead(\
                        y[tslice],T[tslice])
            
        # if i%(24*7) ==0:
        #        print("l: {} b: {}\n".format(str(l),str(b)))
        #        print(s,"\n")
        ypred=pd.Series(ypred,index=y.index)        
        return ypred

    def optimize_param(self,y,T,ninit=4*24*7,rtol=0.01,\
                      eta=0.01,lr=0.2,nmax=100):
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
        smooth_names=['alpha','beta','g00','g01','g10','g11']
        temp_names=['An','Ap','Tn','Tp']
        names=smooth_names+temp_names
        pred0 = self.STL_dayahead(y,T,ninit=ninit)
        J    = self.rmse(y[ninit:],pred0[ninit:])
        Ni=0
        #loop over iterations
        for i in range(nmax):
            dJ_max=0
            lr = lr*0.99
            #for each name, tweak the model's variables.
            for n in names:
                p0=self.__getattribute__(n)           
                self.__setattr__(n,p0*(1+eta))
                pred=self.STL_dayahead(y,T,ninit=ninit)
                J2=self.rmse(y[ninit:],pred[ninit:])
                dJ = (J2-J)/J
                #actually update 
                p = p0-lr*dJ/(eta)
                #restrict smoothing parameters to be within [0,1].
                p=self.check_limits(n,p,p0)
                self.__setattr__(n,p)
                J=J2
                dJ_max=max(dJ,dJ_max)
            Ni+=1       
            if (dJ_max<rtol):
                print("Hit tolerance {} at iter {}".format(dJ,Ni))
                plot_pred([pred,y],['Predicted','Actual'])              
                return pred
            if(Ni%10==0):
                print("Cost, Old Cost = {},{}".format(J,J2))
                plot_pred([pred,y],['Predicted','Actual'])
            print('Iter {}.  Cost {}'.format(Ni,J))
            for n in names:
                p0=self.__getattribute__(n)
                print(n,p0)

        print("Failed to hit tolerance after {} iter\n".format(iter))
        print("Cost:",J,J2)
        return pred 

    def check_limits(self,n,p,p0):
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
