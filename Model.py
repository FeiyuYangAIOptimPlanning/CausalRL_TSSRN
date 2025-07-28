
import torch as th
from torch import nn
from myCausalInferenceModule import CausalInferenceModule
from common.utils import add_noise_to_logits

class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidden_size, output_size, output_act, index, device,z_dim):
        super(ActorNetwork, self).__init__()
        self.index=index
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        #self.dropout=nn.Dropout(0.3)
        # activation function for the output
        self.output_act = output_act

        #causal inference mudule
        #构造图并获取邻接矩阵
        edges=[((0, 1), (1, 2)), ((1, 2), (2, 3)), ((2, 3), (3, 4)), ((3, 4), (4, 5)), ((3, 4), (4, 13)), ((3, 4), (4, 14)), ((4, 5), (4, 13)), ((4, 5), (4, 14)), ((4, 5), (5, 6)), ((4, 13), (4, 14)), ((4, 13), (12, 13)), ((4, 14), (14, 15)), ((5, 6), (6, 7)), ((6, 7), (7, 8)), ((7, 8), (8, 9)), ((10, 11), (11, 12)), ((11, 12), (12, 13)), ((14, 15), (15, 16)), ((15, 16), (16, 17)), ((16, 17), (17, 18))]
        keyIndex=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 13), (4, 14), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16), (16, 17), (17, 18)]        
        self.device=device
        edges_agent_index=[]
        for k,v in edges:
           kindex=keyIndex.index(k)
           vindex=keyIndex.index(v)
           edges_agent_index.append((kindex,vindex))
        edge_index = th.tensor(edges_agent_index, dtype=th.long).t().contiguous().to(self.device)
        self.causal_module=CausalInferenceModule(edge_index,18,self.device,self.index,z_dim)
       
#[7, 7, 7, 10, 10, 10, 10, 7, 7, 7, 9, 7, 7, 7, 7, 7, 7, 9]
    def forward(self, state,isdetachz=False):
        A,B,C,D,E=state
        if self.training==False:
            a=[atmp.to('cpu') for atmp in A]
            b=B.to('cpu')
            d=D.to('cpu')
            c=[ctmp.to('cpu') for ctmp in C]
            e=[etmp.to('cpu') for etmp in E]
        else:
            a=[atmp.to('cuda') for atmp in A]
            b=B.to('cuda')
            d=D.to('cuda')
            c=[ctmp.to('cuda') for ctmp in C]
            e=[etmp.to('cuda') for etmp in E]
        zst,zst_for_critic,vaelossinput,predilossinput,predirlossinput=self.causal_module(a,b,c,d)
        
        out = nn.functional.tanh(self.fc1(zst))
        out=self.fc2(out)
        # 应用动作掩码
        masked_logits = th.where(e[self.index]!=0, out, th.tensor(-100.0).to(out.device))#th.where(condition?,ifTrueReturn,ifFalseReturn)
        masked_logits_withNoise=add_noise_to_logits(masked_logits)
        log_action_probs = self.output_act(masked_logits_withNoise,dim=-1)#self.output_act=nn.functional.log_softmax
        return log_action_probs, zst_for_critic, vaelossinput, predilossinput, predirlossinput,e[self.index]#e[self.index]==0->masked


class CriticNetwork(nn.Module):
    """
    A network for critic
    """
    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], -1)#Edited 1 -> -1
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# class ActorCriticNetwork(nn.Module):
#     """
#     An actor-critic network that shared lower-layer representations but
#     have distinct output layers
#     """
#     def __init__(self, state_dim, action_dim, hidden_size,
#                  actor_output_act, critic_output_size=1):
#         super(ActorCriticNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.actor_linear = nn.Linear(hidden_size, action_dim)
#         self.critic_linear = nn.Linear(hidden_size, critic_output_size)
#         self.actor_output_act = actor_output_act

#     def forward(self, state):
#         out = nn.functional.relu(self.fc1(state))
#         out = nn.functional.relu(self.fc2(out))
#         act = self.actor_output_act(self.actor_linear(out))
#         val = self.critic_linear(out)
#         return act, val