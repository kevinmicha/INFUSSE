# -*- coding: utf-8 -*-

r"""This module contains the model class. 

:Authors:   Kevin Michalewicz <k.michalewicz22@imperial.ac.uk>

"""

import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, lm_dim=1024, lm=None):
        super().__init__()
        self.lm_dim = lm_dim
        self.lm = lm
        self.aa_linear = torch.nn.Linear(in_channels, self.lm_dim, bias=False)
        self.c_linear = torch.nn.Linear(in_channels, self.lm_dim, bias=False)
        self.lm_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        #self.ab_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)
        #self.ag_linear = torch.nn.Linear(self.lm_dim, self.lm_dim, bias=True)

        self.conv1 = GCNConv(self.lm_dim, hidden_channels, normalize=True)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels, normalize=True)

        self.conv3 = GCNConv(hidden_channels, out_channels, normalize=True)

    def forward(self, x, x_out, edge_index, edge_weight=None, c=None):#, len_ab=None, len_ag=0):
        #batch_size = len(len_ag)
        if self.lm_dim != 1024:
            x = self.lm(x)[0]
            x = (self.aa_linear(x) + self.lm_linear(x)).relu()
        else:
            #print(x.shape)
            #mask = x > 4
            #x_out = self.lm(x[None,:].to(torch.int64), output_attentions=False, output_hidden_states=True)['hidden_states'][-1]
            #x_out = x_out[mask.unsqueeze(-1).expand_as(x_out)].view(x_out.size(0), -1, x_out.size(-1))
            #x = x[mask].to(torch.float32).unsqueeze(-1)
            x = (self.c_linear(c.unsqueeze(-1)) + self.aa_linear(x.unsqueeze(-1)) + self.lm_linear(x_out)).relu()

            '''
            x_processed = []
            start = 0 

            for i in range(batch_size):
                ab_len = len_ab[i]   
                ag_len = len_ag[i] if len_ag is not None else 0 
                
                x_ab = x[start:start + ab_len]
                x_ab = self.ab_linear(x_ab).relu()  

                if ag_len > 0:
                    x_ag = x[start + ab_len:start + ab_len + ag_len]
                    x_ag = self.ag_linear(x_ag).relu()  
                    x_combined = torch.cat((x_ab, x_ag), dim=0) 
                else:
                    x_combined = x_ab  

                x_processed.append(x_combined)
                
                start += ab_len + ag_len

            x = torch.cat(x_processed, dim=0)
            '''
        x = self.conv1(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        #x = self.conv2(x, torch.squeeze(edge_index), torch.squeeze(edge_weight)).relu()
        x = self.conv3(x, torch.squeeze(edge_index), torch.squeeze(edge_weight))

        return x