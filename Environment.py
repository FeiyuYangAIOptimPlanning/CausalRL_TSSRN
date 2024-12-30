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
from SimulatorClasses import SimulatorBasedLinesAndTrains,ResponseToTheSchedulingAction,SimulatorStatu,RailTrain,SchedulingAction
import pickle
from datetime import datetime
from common.utils import addScalarFcn,globalEpisode
class Environment:
    FixedLineParams={
    'stationNamesList':['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','B1','B2','B3','B4','B6','B7','B8','B9','B10'],
    'stationTrackCntList':[3,3,3,3,6,3,3,3,3,5,3,3,3,3,3,3,3,3,5],
    'sectionIndexList':[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],
                    [10,11],[11,12],[12,13],[13,4],[4,14],[14,15],[15,16],[16,17],[17,18]],
    'sectionLengths':[10000,20000,10000,20000,20000,10000,20000,20000,20000,20000,20000,20000,10000,20000,20000,10000,20000,10000],
    }

    LinePlanParams={
    'trainCntLine1':8,
    'trainCntLine2':8,
    'totalTrainCnt':8+8,
    'initTimeList':[0,40,80,120,160,200,240,280,0,40,80,120,160,200,240,280],
    'oStaIndexList':[0,9,0,9,0,9,0,9,10,18,10,18,10,18,10,18],
    'dStaIndexList':[9,0,9,0,9,0,9,0,18,10,18,10,18,10,18,10],
    'velocityList':[1500,1000,500,1500,1500,1500,1500,1500,1500,1000,500,1500,1500,1500,1500,1500],
    'priorityList':[3,2,1,3,3,3,3,3,3,2,1,3,3,3,3,3]
    }
  
    
    
    def __init__(self,defaultLinePlanParams=LinePlanParams):
        self.simulator=SimulatorBasedLinesAndTrains(FixedLineParams=Environment.FixedLineParams,LinePlanParams=defaultLinePlanParams)
        self.isValidLinePlanParams,_,self.state=self.simulator.Start()
        if not self.isValidLinePlanParams:
            print('The Line Plan Params causes a invalid line plan')
            return None
        self.sections=self.simulator.lines.sections
        
        self.sectionCount=len(self.sections)
        self.rewardBySections=None
        
    def reset(self):
        self.simulator.Reset()
        return self.simulator.Start()
    
    def step(self,envSchedulingActions):
      
        self.simulator.resetTheRewards()
        railTrain_schedulingAction_response_tup_list=self.simulator.ReceiveAndExecuteDecisions(envSchedulingActions)
        
        res=set([rsr[2] for rsr in railTrain_schedulingAction_response_tup_list])
        if ResponseToTheSchedulingAction.ValidActionWithFailExecuting in res or self.simulator.statu==SimulatorStatu.CompletedWithFailure:
    
            isDone=True
            isSuccess=False
            print(f'    Env:eposide end, isDone {isDone}, isSuccess {isSuccess}')
            return self.simulator.State(), self.GetCurrentTotalRewardbyEachAgent(True),[v for k,v in self.simulator.sectionDoneDicts.items()],False,self.ActionMasks(False)
            
        timeStepSuccess=self.simulator.TimeStepUpdate()
        if not timeStepSuccess:
            
            isDone=True
            isSuccess=False
            print(f'    Env:eposide end, isDone {isDone}, isSuccess {isSuccess}')
            return self.simulator.State(), self.GetCurrentTotalRewardbyEachAgent(True),[v for k,v in self.simulator.sectionDoneDicts.items()],False,self.ActionMasks(False)
       
        isDone,isSuccess,isoutoftime=self.simulator.IsDoneAndSimStatu()
        if isoutoftime:
            print(f'        Env:Out of time')
        if isDone:
            print(f'    Env:eposide end, isDone {isDone}, isSuccess {isSuccess}********')
            addScalarFcn('PW_delay',self.simulator.pd,globalstep=globalEpisode[0])
            if self.simulator.pd<=SimulatorBasedLinesAndTrains.MinPD and self.simulator.pd>0:
        
                addScalarFcn('Min_PW_delay',self.simulator.pd,globalstep=globalEpisode[0])
                SimulatorBasedLinesAndTrains.MinPD=self.simulator.pd
                now = datetime.now()
                date_time_string = now.strftime('%Y-%m-%d-%H-%M')
                with open(f'success_simulator_{date_time_string}_pd_{self.simulator.pd}.pkl', 'wb') as file:
                    pickle.dump(self.simulator, file)
            return self.simulator.State(), self.GetCurrentTotalRewardbyEachAgent(True),[v for k,v in self.simulator.sectionDoneDicts.items()],True,self.ActionMasks(True)
        nextState=self.simulator.State()
        return nextState, self.GetCurrentTotalRewardbyEachAgent(False),[v for k,v in self.simulator.sectionDoneDicts.items()], True , self.ActionMasks(True)
        
    def GetCurrentTotalRewardbyEachAgent(self,isdone):
        section_direct_Rs=[]
        for section in self.sections:
            sectionR=self.simulator.sectionRewardDicts[section]
            section_neighbor1=self.simulator.stationRewardDicts[section.station1]
            section_neighbor2=self.simulator.stationRewardDicts[section.station2]
            section_direct_Rs.append(0.5*sectionR+0.25*section_neighbor1+0.25*section_neighbor2)
        for idx,section in enumerate(self.sections):
            sectionR=section_direct_Rs[idx]
            station1=section.station1
            station2=section.station2
            self.simulator.stationRewardDicts[station1]+=0.25*sectionR
            self.simulator.stationRewardDicts[station2]+=0.25*sectionR
        for idx,section in enumerate(self.sections):
            sectionR=section_direct_Rs[idx]
            section_neighbor1=self.simulator.stationRewardDicts[section.station1]
            section_neighbor2=self.simulator.stationRewardDicts[section.station2]
            section_direct_R=sectionR+0.1*section_neighbor1+0.1*section_neighbor2 
            section_direct_Rs[idx]=section_direct_R
        self.rewardBySections=[sdr for sdr in section_direct_Rs]
        return self.rewardBySections
    
    def GetEnvActions(self,section,actionIndex):
        """
        
        """
        selectedTrack=None
        tracks=[]
        for track in section.station1.tracks:
            tracks.append(track)
        for track in section.station2.tracks:
            tracks.append(track)
        if actionIndex==len(tracks):
            selectedTrack=None
        else:
            selectedTrack= tracks[actionIndex]
        decisionTupleList=self.ConvertActionToSchedulingActions(section,selectedTrack,tracks)
        return decisionTupleList
     
    def ConvertActionToSchedulingActions(self, section, selectedTrack, tracks):
        """
        
        """
        trains=[track.trainRef for track in tracks if isinstance(track.trainRef,RailTrain)]
        trainDecisions=[]
        for train in trains:
            currentResourceIndex=train.shortestPathInGraph.index(train.currentNode)
            if currentResourceIndex==len(train.shortestPathInGraph)-1:
                trainDecisions.append((train,SchedulingAction.MoveAction))
            else:
                if(train.shortestPathInGraph[currentResourceIndex+1]['sectionObjRef'] is section):
                    if train.GetTheTrack() is selectedTrack:
                        if self.simulator.lines.isTrainValidToMove(train):
                            trainDecisions.append((train,SchedulingAction.MoveAction))
                        else:
                            self.simulator.sectionRewardDicts[section]+=-self.simulator.failedBase*train.priority
                            trainDecisions.append((train,SchedulingAction.DwellAction))
                
                    else:
                        trainDecisions.append((train,SchedulingAction.DwellAction))
                else:
                    self.simulator.sectionRewardDicts[section]+=-self.simulator.invalidActionBase
        if section.tracks[0].trainRef is not None:
            trainDecisions.append((section.tracks[0].trainRef,SchedulingAction.MaintainTheStatu))
        decisionTupleList=trainDecisions
        return decisionTupleList
    
    def ActionMasks(self,isfailed):
        legal_actions_mask=[[False for _ in range(len(section.station1.tracks+section.station2.tracks)+1)] for section in self.sections]
        if not isfailed:
            return legal_actions_mask
        for secidx,section in enumerate(self.sections):
            tracks=[]
            for track in section.station1.tracks:
                tracks.append(track)
            for track in section.station2.tracks:
                tracks.append(track)
            for trackidx,sectrack in enumerate(tracks):
                if sectrack.trainRef == None:
                    legal_actions_mask[secidx][trackidx]=False
                else:
                    train=sectrack.trainRef
                    sectionForward=train.NextSource()
                    if sectionForward is not None:
                        if sectionForward['sectionObjRef'] is section and self.simulator.lines.isTrainValidToMove(train)==True:
                            legal_actions_mask[secidx][trackidx]=True
                    else:
                        legal_actions_mask[secidx][trackidx]=True
            legal_actions_mask[secidx][-1]=True
        return legal_actions_mask
        


