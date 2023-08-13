import torch
import torch.nn as nn

class GRUcell(nn.Module):
    def __init__(self, input, input_embedding, hidden_state, f_bias=1.0, L2=False, h_act=nn.Tanh, init_h=None, init_c=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.input = input.to(self.device)
        self.input_embedding = input_embedding
        self.hidden_state = hidden_state
        self.type = "gru"
        if init_h is None:
            self.init_h = torch.matmul(self.input[0,:,:], torch.zeros([self.input_embedding, self.hidden_state]))
            self.previous = self.init_h # (1, input_embedding, hidden_state) 
        self.r = self.gate()
        self.u = self.gate()
        self.cell = self.gate()
        # self.distance = torch.tensor([torch.sin(torch.tensor(i * torch.pi/2)) for i in range(4)])
        self.W_in = torch.nn.init.orthogonal(torch.FloatTensor(self.input_embedding, self.hidden_state)) # (50,50)
        self.b_in = torch.full([self.hidden_state], 0.001) # (50)
        self.W_x = torch.cat((self.r[0], self.u[0], self.cell[0]), dim=1)  # (50,150)
        self.W_h = torch.cat((self.r[1], self.u[1], self.cell[1]), dim=1)  # (50,150)
        if L2:
            self.L2_loss = self.L2_loss(self.W_x) + self.L2_loss(self.W_h) + self.L2_loss(self.W_in)
    
    def L2_loss(self,x):
        return torch.abs(torch.sum(torch.pow(x,2)))
    
    def gate(self):
        W_x = torch.nn.init.orthogonal(torch.FloatTensor(self.input_embedding, self.hidden_state)) # (50,50)
        W_h = torch.nn.init.orthogonal(torch.FloatTensor(self.hidden_state, self.hidden_state)) # (50,50)
        return W_x, W_h
    
    def slice_w(self, x, n):
        return x[:, n*self.hidden_state:(n+1)*self.hidden_state]

    def step(self, prev_h, current_x, U_d):
        node_f = torch.sigmoid(torch.matmul(current_x,self.W_in) + self.b_in) # node_f = sigmoid(W_in * X + b_in) (12,50)
        distance_inf = node_f * U_d # (12,50) × (12,1) = (12,50)
        distance_inf = torch.cat((distance_inf,distance_inf,distance_inf),dim=1) # (12,150)
        W_x = torch.matmul(current_x, self.W_x) + distance_inf # (12,150)
        W_h = torch.matmul(prev_h, self.W_h) # (12,150)

        # reset gate
        r = torch.sigmoid(self.slice_w(W_x, 0) + self.slice_w(W_h, 0))
        # update gate
        u = torch.sigmoid(self.slice_w(W_x, 1) + self.slice_w(W_h, 1))
        c = torch.tanh(self.slice_w(W_x, 2) + r * self.slice_w(W_h, 2))
        current_h = (1-u) * prev_h + u * c
        
        return current_h # (12,50)

def RNN(cell, cell_b=None, merge="sum", U_d=None):

    h_states = []
    for step in range(cell.input.shape[0]): 
        current_h = cell.step(cell.previous, cell.input[step,:,:], U_d[step,:,:])  
        cell.previous = current_h
        h_states.append(current_h)
    h_states = torch.stack(h_states,dim=0) # (5,12,50)

    if cell_b is not None:
        input_b = torch.flip(cell.input, dims=[0]) 
        U_d_b = torch.flip(U_d, dims=[0])
        h_states_b_rev = []
        for step in range(input_b.shape[0]):
            current_h_b = cell_b.step(cell_b.previous, input_b[step,:,:], U_d_b[step,:,:])  
            cell_b.previous = current_h_b
            h_states_b_rev.append(current_h_b)
        h_states_b_rev = torch.stack(h_states_b_rev,dim=0) # (5,12,50)

        h_states_b = torch.flip(h_states_b_rev, dims=[0])
        if merge == "sum":
            h_states = h_states + h_states_b # (5,12,50)
        else:
            h_states = torch.cat((h_states,h_states_b), dim=2) # (5,12,100)

    return h_states   

