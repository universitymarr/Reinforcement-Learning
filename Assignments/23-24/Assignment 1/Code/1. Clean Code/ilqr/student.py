from autograd import grad, jacobian
import autograd.numpy as np


class ILqr:
    
    def __init__(self, dynamics, cost, horizon=50):
        
        self.f = dynamics
        self.horizon = horizon
        
        self.getA = jacobian(self.f,0)
        self.getB = jacobian(self.f,1)

        self.cost = cost
        self.getq = grad(self.cost,0)
        self.getr = grad(self.cost,1)
        
        self.getQ = jacobian(self.getq,0)
        self.getR = jacobian(self.getr,1)
        
    def backward(self, x_seq, u_seq):
        
        pt1 = self.getq(x_seq[-1],u_seq[-1])
        Pt1 = self.getQ(x_seq[-1],u_seq[-1])
        
        k_seq = []
        K_seq = []
        
        for t in range(self.horizon-1,-1,-1):

            xt = x_seq[t]
            ut = u_seq[t]
            
            At = self.getA(xt,ut)
            Bt = self.getB(xt,ut)
            
            qt = self.getq(xt,ut)
            rt = self.getr(xt,ut)
            
            Qt = self.getQ(xt,ut)
            Rt = self.getR(xt,ut)

            # TODO
            kt = ...
            Kt = ...
            # TODO
            pt = ...
            Pt = ...

            pt1 = pt
            Pt1 = Pt

            k_seq.append(kt)
            K_seq.append(Kt)
        
        k_seq.reverse()
        K_seq.reverse()
        
        return k_seq,K_seq
    
    def forward(self, x_seq, u_seq, k_seq, K_seq):
        
        x_seq_hat = np.array(x_seq)
        u_seq_hat = np.array(u_seq)
        
        for t in range(len(u_seq)):
            # TODO
            control = ...
            
            # clip controls to the actual range from gymnasium
            u_seq_hat[t] = np.clip(u_seq[t] + control,-2,2)
            x_seq_hat[t+1] = self.f(x_seq_hat[t], u_seq_hat[t])
            
        return x_seq_hat, u_seq_hat


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi

def cost(x,u):
    costs = angle_normalize(x[0])**2 + .1*x[1]**2 + .001*(u**2)
    return costs

def pendulum_dyn(x,u):    
    th = x[0]
    thdot = x[1]

    g = 10.
    m = 1.
    l = 1.
    dt = 0.05

    u = np.clip(u, -2, 2)[0]

    # TODO
    newthdot = ...
    newth = ...
    
    newthdot = np.clip(newthdot, -8, 8)

    x = np.array([newth, newthdot])
    return x
