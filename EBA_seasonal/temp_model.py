class just_temp_model:
    def __init__(self,names=[],vals=[]):
        self.param= self.param_vec(names,vals)

    def param_vec(self,names,vals):
        return pd.Series(vals,index=names)

    def temp_model(self,T):
        """temp_model(self,T)
        Tries to fit linear model for electricity demand to temperature.
        Initially tried to allow thresholding, and different slopes for heating/cooling.  
        """ 
        m1 = T>self.param['Tp']
        m2 = T<self.param['Tn']
        y=np.zeros(T.shape)
        #now allows Tp<Tn, and adds effects.
        y[m1] = (T[m1]-self.param['Tp'])*self.param['ap']
        y[m2] += (self.param['Tn']-T[m2])*self.param['an']
        y+=+self.param['a0']
        #y = param['a0']+ param['ap']*np.abs(T-param['Tp'])
        y=pd.Series(y,name='Predicted Demand',index=T.index)
        return y

    def temp_model_single(self,T):
        """temp_model(self,T)
        Tries to fit linear model for electricity demand to temperature.
        Initially tried to allow thresholding, and different slopes for heating/cooling.  
        Will now just try simpler linear model a_p|T-T_p|.
        """ 
        #m1 = T>self.param['Tp']
        #m2 = T<self.param['Tn']
        # y=np.zeros(T.shape)
        # y[m1] = (T[m1]-self.param['Tp'])*self.param['ap']
        # y[m2] = (self.param['Tn']-T[m2])*self.param['an']
        # y=y+self.param['a0']
        y = self.param['a0']+ self.param['ap']*np.abs(T-self.param['Tp'])
        y=pd.Series(y,name='Predicted Demand',index=T.index)
        return y


    def temp_model_grad(self,D,Dhat,T):
        """temp_model_grad
        Compute gradients of model w.r.t. parameters.
        Assumes loss-function is mean-square.
        Dhat - measured demand
        D    - predicted demand
        T    - measured temperature
        """
        m1 = T>self.param['Tp']
        m2 = T<self.param['Tn']
        Nt = len(T)
        Derr=D-Dhat
        #initialize with zeros
        dparam =self.param_vec( self.param.keys(), np.zeros(len(self.param)))
        dparam['a0'] = np.sum(Derr)/Nt
        #Single model
        # dparam['ap'] = np.sum( np.abs(T-param['Tp'])*Derr)/Nt
        # dparam['Tp'] = -np.sum(np.sign(T-param['Tp'])*param['ap']*Derr)/Nt
        #Double thresholded model
        dparam['ap'] = np.sum( (T[m1]-self.param['Tp'])*(Derr[m1]))/Nt
        dparam['an'] = np.sum( (self.param['Tn']-T[m2])*(Derr[m2]))/Nt
        dparam['Tp'] = -self.param['ap']*np.sum(Derr[m1])/Nt
        dparam['Tn'] =  self.param['an']*np.sum(Derr[m2])/Nt    
        return dparam
    
    def param_fit(self,Dhat,T,alpha=0.1,rtol=1E-4,nmax=200):
        """Try to fit linear threshold model of demand to temperature.
            D - demand data
            T - temperature data
            Fits model of form:
            D ~ a_0+ a_p[T-T_p]_+ + a_n[T_n-T]_+,
            where [f]_+ =f for f>0, and 0 otherwise.

            Just use simple gradient descent to fit the model.
        """
        #make parameter estimates
        Dr = np.max(Dhat)-np.min(Dhat)
        Tr = np.max(T)-np.min(T)
        param_names=['a0','ap','an','Tp','Tn']
        param_vals=[np.mean(Dhat), 0.5*Dr/Tr, 0.5*Dr/Tr,
        np.mean(T), np.mean(T)]
        self.param=self.param_vec(param_names,param_vals)
        Dpred = self.temp_model(T)
        J=np.sum((Dpred-Dhat)**2)/len(Dhat)
        print('Init cost',J)
        plot_pred([Dpred,Dhat,T],['Predicted','Actual','Temp'])
        print('Param:',self.param,"\n")    

        Ni=0
        for i in range(nmax):
            dparam=self.temp_model_grad(Dpred,Dhat,T)
            self.param=self.param-alpha*dparam
            Dpred=self.temp_model(T)
            J2=J        
            J=np.sum((Dpred-Dhat)**2)/len(Dhat)
            err_change=abs(1-J2/J)
            Ni+=1
            if (err_change<rtol):
               print("Hit tolerance {} at iter {}".format(
               err_change,Ni))
               plot_pred([Dpred,Dhat,T], ['Predicted','Actual','Temp'])
               return Dpred
            if(Ni%100==0):
                print("Cost, Old Cost = {},{}".format(J,J2))
                print('Param:',self.param)
                print('Param_grad:',dparam)
                print("Mean param Change {} at iter {}".format( err_change,Ni))
                plot_pred([Dpred,Dhat,T],['Predicted','Actual','Temp'])
        print("Failed to hit tolerance after {} iter\n".format(iter))
        print("Cost:",J,J2)
        return Dpred 

    def grad_check(self,Dhat,T):
        """grad_check(Dhat,T)
        Check numerical gradients against finite difference.
            D - demand data
            T - temperature data
        """
        #make parameter estimates
        Dr = np.max(Dhat)-np.min(Dhat)
        Tr = np.max(T)-np.min(T)
        param_names=['a0','ap','an','Tp','Tn']
        #param_vals=300*np.random.random(size=5)
        param_vals=[np.mean(D),5*Dr/Tr,5*Dr/Tr,150,200]        
        self.param=self.param_vec(param_names,param_vals)
        Dpred = self.temp_model(T)    
        dparam=self.temp_model_grad(Dpred,Dhat,T)    
        J=0.5*np.sum((Dpred-Dhat)**2)/len(Dhat)
        eps=.001
        param2 = self.param.copy()
        print('Name','Num Grad','Anlt Grad', 'Param')
        for name,val in param2.items():
            self.param = param2.copy()
            self.param[name]=val+eps
            Dpred = self.temp_model(T)
            J2=0.5*np.sum((Dpred-Dhat)**2)/len(Dhat)
            self.param[name]=val
            numgrad = (J2-J)/eps
            print(name,numgrad,dparam[name],self.param[name])

#End of temp_model class.
#Needed to scale the data to get decent answers.  
