# MIT License

# Copyright (c) 2024 Feiyu Yang 杨飞宇

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
import numpy as np
from Environment import Environment
import torch
from gym import spaces
from myTools import action2code

class MyEnvironmentRay():
    def __init__(self,device, num_agents=18):
        super().__init__()
        self.CIMode=True
        self.num_agents = num_agents
        edges=[((0, 1), (1, 2)), ((1, 2), (2, 3)), ((2, 3), (3, 4)), ((3, 4), (4, 5)), ((3, 4), (4, 13)), ((3, 4), (4, 14)), ((4, 5), (4, 13)), ((4, 5), (4, 14)), ((4, 5), (5, 6)), ((4, 13), (4, 14)), ((4, 13), (12, 13)), ((4, 14), (14, 15)), ((5, 6), (6, 7)), ((6, 7), (7, 8)), ((7, 8), (8, 9)), ((10, 11), (11, 12)), ((11, 12), (12, 13)), ((14, 15), (15, 16)), ((15, 16), (16, 17)), ((16, 17), (17, 18))]
        keyIndex=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 13), (4, 14), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16), (16, 17), (17, 18)]
        self.env=Environment()
        self.orderedSections=[] 
        #env sim record
        self.episode=0
        self.ep_step=0
        self.trainingIndex=1
        self.device=device

        for secIndex in keyIndex:
            for section in self.env.sections:
                k,v=secIndex
                if section.sectionIndex==f'{k}-{v}' or section.sectionIndex==f'{v}-{k}':
                    self.orderedSections.append(section)
        self.observation_space_dims = [58,58,58,82,82,82,82,58,58,58,74,58,58,58,58,58,58,74]
        self.action_space_dims = [len(section.station1.tracks)+len(section.station2.tracks)+1 for section in self.orderedSections]
                
        self.trajectoryByAgent=None
        self.current_self_obv_ByAgent=[] 
       
    def reset(self, seed=None, options=None):
       
        _,_,_=self.env.reset()
        self.episode+=1
        self.ep_step=0

        self.trajectoryByAgent=[]#
        self.current_self_obv_ByAgent=[np.array(section.NormalizationObvEncodeing(maxVelocity=1500,thTimeValue=20),dtype=np.float32) for section in self.orderedSections]#每一步结束时清空，凑齐sars'自动清空,每一幕开始手动赋初始值
     
        tranjectorysLastTime=[]
        for secindex,section in enumerate(self.orderedSections):
        
            presarTuple=(np.zeros_like(section.NormalizationObvEncodeing(maxVelocity=1500,thTimeValue=20),dtype=np.float32),np.array([self.action_space_dims[secindex]],dtype=np.float32),np.array([0],dtype=np.float32))
            tranjectorysLastTime.append(presarTuple)
        self.trajectoryByAgent.append(tranjectorysLastTime)
        stateList=[]
        for idx,obvByAgent in enumerate(self.current_self_obv_ByAgent):
            stateList.append(obvByAgent[np.newaxis, :])
        self.state=np.concatenate(stateList,axis=-1)
     
        A = [tra[0] for tra in self.trajectoryByAgent[-1]]
        B = torch.stack([action2code(int(tra[1][0])) for tra in self.trajectoryByAgent[-1]]).numpy()
        C = [rawObv for rawObv in self.current_self_obv_ByAgent]
        D = torch.stack([torch.tensor(tra[2], dtype=torch.float32) for tra in self.trajectoryByAgent[-1]]).numpy()
        return (A,B,C,D,self.env.ActionMasks(True))

    def step(self, action_dict):
        """根据智能体的动作字典来更新环境状态，并返回下一步的观察、奖励、完成标志和额外信息。"""
        if isinstance(action_dict,list):
            action_dict={idx:act for idx,act in enumerate(action_dict)}
        preRawObvsList=[np.array(section.NormalizationObvEncodeing(maxVelocity=1500,thTimeValue=20),dtype=np.float32) for section in self.orderedSections]
        railTrain_schedulingAction_Tup_List=[]
        for agent_id, action in action_dict.items():
            decisionTupleList=self.env.GetEnvActions(section=self.orderedSections[agent_id],actionIndex=action)
            railTrain_schedulingAction_Tup_List.extend(decisionTupleList)
        _, rewards, dones, _,nextActionMasks = self.env.step(railTrain_schedulingAction_Tup_List)
        self.ep_step+=1
        self.env.simulator.resetTheRewards()

        self.env.simulator.UpdateTrainIsValidMove()
        total_done=max(dones)
        nextRawObvs=[np.array(section.NormalizationObvEncodeing(maxVelocity=1500,thTimeValue=20),dtype=np.float32) for section in self.orderedSections]        
        rewards = {agent_id: np.array(np.float32(rewards[agent_id]),dtype=np.float32) for agent_id in range(self.num_agents)} 
        terminateds = {agent_id: bool(dones[agent_id]) for agent_id in range(self.num_agents)}  
   
        self.current_self_obv_ByAgent=nextRawObvs
        tranjectorysLastTime=[]
        for secindex,section in enumerate(self.orderedSections):
            presarTuple=(preRawObvsList[secindex],np.array([action_dict[secindex]],dtype=np.float32),np.array([rewards[secindex]],dtype=np.float32))
            tranjectorysLastTime.append(presarTuple)
           
        self.trajectoryByAgent.append(tranjectorysLastTime)
        stateList=[]
        for idx,obvByAgent in enumerate(self.current_self_obv_ByAgent):
            stateList.append(obvByAgent[np.newaxis, :])
        self.state=np.concatenate(stateList,axis=-1)
        
        A = [tra[0] for tra in self.trajectoryByAgent[-1]]
        B = torch.stack([action2code(int(tra[1][0])) for tra in self.trajectoryByAgent[-1]]).numpy()
        C = [rawObv for rawObv in self.current_self_obv_ByAgent]
        D = torch.stack([torch.tensor(tra[2], dtype=torch.float32) for tra in self.trajectoryByAgent[-1]]).numpy()
        E = nextActionMasks
      
        truncateds = {agent_id: bool(dones[agent_id]) for agent_id in range(self.num_agents)}
     
        infos = {agent_id: {} for agent_id in range(self.num_agents)}

        return (A,B,C,D,E), [reward for agent,reward in rewards.items()], [done for done in terminateds.values()], [done for done in truncateds.values()],infos
    