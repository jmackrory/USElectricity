class multiseasonal:

    def fit_init_params(y,ninit=4*24*7):
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
        l = np.mean(yval)
        ##average shift
        b = np.mean(np.diff(yval))
        ##remove mean pattern, subtractin off level, and linear trend.    
        ysub = ysub-l-b*np.arange(ninit)
        #mean seasonal pattern.
        #second seasonal pattern is for weekends, with days
        #Saturday/Sunday have dayofweek equal to 5 and 6.
        #make a mask to select out weekends.
        s2 = ysub.index.dayofweek >=5
        #select out weekends, and regular days. 
        y_end = ysub[s2]
        y_week=ysub[~s2]
        n1 = int(len(y_week)/24)
        n2 = int(len(y_end)/24)
        s = np.zeros((2,24))
        print(n1,n2)
        for n in range(n1):
            s[0,:] = s[0,:]+y_week[n*24:(n+1)*24]/n1
        for n in range(n2):
            s[1,:] = s[1,:]+y_end[n*24:(n+1)*24]/n2

        return l, b, s

    # def predict_stl(l,b,s,timeIndex):
    #     """predict_stl(l,b,s,timeIndex)
    #     Predicts STL time-series for a fixed set of parameters.
    #     Not useful.
    #     """
    #     # n1 = int(sum(~msk)/24)    
    #     # n2 = int(sum(msk)/24)
    #     #Use fact that first sub-season is weekdays in first row.
    #     #Use integer conversion of true/false to 0/1.
    #     #Then use fact that seasonal patterns are 24 hours long to select right hour.
    #     #find weekend/weekedays.  
    #     msk=timeIndex.dayofweek>=5
    #     trend=l+b*np.arange(len(timeIndex))
    #     pred=trend+s[msk.astype(int),timeIndex.hour.values]
    #     return pred

    def STL_step(l,b,s,alpha,beta,gamma,yhat,ypred,t):
        """STL_step
        Updates time parameters based on differences between predicted and 
        measured.  (Not fixing the model parameters for update strength!)
        Work in Progress
        """
        eps=(yhat-ypred)
        #find seasonal patterns.  
        m1 = t.hour                   
        mr = int(t.dayofweek>=5)
        ynew = l + b + s[mr,m1]
        l = l+b+alpha*eps
        b = b+beta*eps
        #update row, and hour
        ds =np.dot(gamma,np.array([~mr,mr]))*eps
        s[:,m1] += ds
        # global icount
        # icount +=1
        # if (icount%(24*7)==0):
        #     print(t)
        #     print(alpha*eps,beta*eps,ds,'\n')
        return l,b,s,ynew

    def predict_STL(y,alpha,beta,gamma):
        """predict_STL
        Generates initial parameters, and then predicts remainder
        of series for input data y.  
        """
    
        ninit=4*24*7
        l,b,s=fit_init_params(y,ninit=4*24*7)
        t0=y.index[ninit+1]
        m1 = t0.dayofweek>=5
        m2 = t0.hour
    
        ypred = l+b*ninit+s[int(m1),m2]
        ytot = np.zeros(len(y))
        print(s[int(m1),m2])
        for i in range(ninit,len(y)):
        l,b,s,ypred = STL_step(l,b,s,
                            alpha,beta,gamma,
                             y[i],ypred,y.index[i])
        ytot[i]=ypred
        # if i%(24*7) ==0:
        #    print("l: {} b: {}\n".format(str(l),str(b)))
        #    print(s,"\n")
        ytot=pd.Series(ytot,index=y.index)        
        return ytot,l,b,s
