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
import networkx as nx
import matplotlib.pyplot as plt
import copy
from enum import Enum,auto
import math
import random

class StationTrack:
    def __init__(self,trackIndex):
        #attributes
        self.trackIndex=trackIndex
        self.tauLeft=0
        self.tauRight=0
        self.tauDwell=0
        #ref info
        self.trainRef=None
    def reset(self):
        self.tauLeft=0
        self.tauRight=0
        self.tauDwell=0
        self.trainRef=None
    def NormalizationObvEncodeing(self,maxVelocity,thTimeValue):
        """
        return {'tauLeft':tauLeft/thTimeValue,'tauRight':tauRight/thTimeValue,'tauDwell':tauDwell/thTimeValue,'trainPriority':trainPriority/3,'trainVelocity':trainVelocity/maxVelocity,'trainDirect':trainDirect/1}
        """
        tauLeft=min(self.tauLeft,thTimeValue)
        tauRight=min(self.tauRight,thTimeValue)
        tauDwell=DwellingTimeSigmoid(self.tauDwell)
        trainPriority=0
        trainVelocity=0
        trainDirect=0
        trainStatu=0
        trainIsValid=0 
       
        if isinstance(self.trainRef,RailTrain):
            trainPriority=self.trainRef.priority
            trainVelocity=self.trainRef.velocity
            trainDirect=1
            trainStatu=self.trainRef.statu.value
            if self.trainRef.isValidMove is None:
                trainIsValid=random.random()
            else:
                if self.trainRef.isValidMove==True:
                    trainIsValid=0.25
                else:
                    trainIsValid=0.75
        stationTrackCode=[tauLeft/thTimeValue,tauDwell,trainPriority/3,trainVelocity/maxVelocity,trainDirect/1,trainStatu/(len(list(RailTrainStatu))-1),trainIsValid,tauRight/thTimeValue]
        assert max(stationTrackCode)<=1,f'exist value is bigger than 1'
        assert min(stationTrackCode)>=0,f'exist value is lower than 0'
        return stationTrackCode

    def GetTheStationTrackLeftTimer(self):
        return self.tauLeft
    def GetTheStationTrackRightTimer(self):
        return self.tauRight
    def SetTheStationTrackLeftTimer(self,newCnt):
        self.tauLeft=max(self.tauLeft,newCnt)
    def SetTheStationTrackRightTimer(self,newCnt):
        self.tauRight=max(self.tauRight,newCnt)
    def GetTheStationTrackDwellTimer(self):
        return self.tauDwell
    def SetTheStationTrackDwellTimer(self,newCnt):
        self.tauDwell=max(self.tauDwell,newCnt)

class Station:
    def __init__(self,stationIndex,name,trackCnt):
        self.stationIndex=stationIndex
        self.name=name
        self.tracks=[StationTrack(i) for i in range(0,trackCnt)]
        self.tauLeft=0
        self.tauRight=0
        self.delta=trackCnt
    def reset(self):
        for track in self.tracks:
            track.reset()
        self.tauLeft=0
        self.tauRight=0
        self.delta=len(self.tracks)
    def NormalizationObvEncodeing(self,maxVelocity,thTimeValue):
        """
       
        """
        tauLeft=min(self.tauLeft,thTimeValue)
        tauRight=min(self.tauRight,thTimeValue)
        delta=len([track for track in self.tracks if track.trainRef is None])
        stationCode=[]
        stationCode.append(tauLeft/thTimeValue)
        tracksCode=[]
        for track in self.tracks:
            tracksCode.extend(track.NormalizationObvEncodeing(maxVelocity,thTimeValue))
        stationCode.extend(tracksCode)
        stationCode.append(tauRight/thTimeValue)
        return stationCode
    def GetTheStationLeftTimer(self):
        return self.tauLeft
    def GetTheStationRightTimer(self):
        return self.tauRight
    def SetTheStationLeftTimer(self,newCnt):
        self.tauLeft=max(self.tauLeft,newCnt)
        for track in self.tracks:
            track.tauLeft=max(track.tauLeft,self.tauLeft)
    def SetTheStationRightTimer(self,newCnt):
        self.tauRight=max(self.tauRight,newCnt)
        for track in self.tracks:
            track.tauRight=max(track.tauRight,self.tauRight)
    def FindValidTracksFromLeftside(self):
        if self.GetTheStationLeftTimer()>0:
            return None
        else:
            trackTmp=None
            for track in self.tracks:
                if track.tauLeft==0 and track.trainRef is None:
                    trackTmp=track
                    return trackTmp
            return None
    def FindValidTracksFromRightside(self):
        if self.GetTheStationRightTimer()>0:
            return None
        else:
            trackTmp=None
            for track in self.tracks:
                if track.tauRight==0 and track.trainRef is None:
                    trackTmp=track
                    return trackTmp
            return None
    def StartTrainFromLeftside(self,railTrain,tau):
        """
        """
        track=self.FindValidTracksFromLeftside()
        if track is None:
            return False,None,self
        else:
            track.trainRef=railTrain
            track.SetTheStationTrackLeftTimer(Lines.TAUA2A)
            track.SetTheStationTrackRightTimer(Lines.TAUA2A)
            track.SetTheStationTrackDwellTimer(0)
            self.SetTheStationLeftTimer(Lines.TAUA2A)
            self.SetTheStationRightTimer(Lines.TAUA2A)
            railTrain.statu=RailTrainStatu.InOperation
            railTrain.currentNode=railTrain.shortestPathInGraph[0]
            railTrain.currentEdge=None
            railTrain.trRecord.append((tau,self.stationIndex,track.trackIndex,SchedulingAction.InitAction,railTrain.statu))
            railTrain.actionToExecute=SchedulingAction.NeedDecision
            return True,track,self
    def StartTrainFromRightside(self,railTrain,tau):
        """
        
        """
        track=self.FindValidTracksFromRightside()
        if track is None:
            return False,None,self
        else:
            track.trainRef=railTrain
            track.SetTheStationTrackLeftTimer(Lines.TAUA2A)
            track.SetTheStationTrackRightTimer(Lines.TAUA2A)
            track.SetTheStationTrackDwellTimer(0)
            
            self.SetTheStationLeftTimer(Lines.TAUA2A)
            self.SetTheStationRightTimer(Lines.TAUA2A)
           
            railTrain.statu=RailTrainStatu.InOperation
            railTrain.currentNode=railTrain.shortestPathInGraph[0]
            railTrain.currentEdge=None
            railTrain.trRecord.append((tau,self.stationIndex,track.trackIndex,SchedulingAction.InitAction,railTrain.statu))
            railTrain.actionToExecute=SchedulingAction.NeedDecision
            return True,track,self   

def DwellingTimeSigmoid(dwellingTime):
    dwellTimeTmp=dwellingTime
    if dwellingTime==0:
        return 0
    elif dwellingTime<Lines.TAUMINDWELL:
        dwellTimeTmp= dwellingTime+1
    else:
        dwellTimeTmp+=3
    if dwellTimeTmp>60:
        dwellTimeTmp=60
    return 2*((2.732**(dwellTimeTmp/2))/(2.732**(dwellTimeTmp/2)+1)-0.5)

class SectionTrack:
    def __init__(self,trackIndex,length):
        #attributes
        self.sIndex=trackIndex
        self.tauLeft=0
        self.tauRight=0
        self.secLength=length
        self.remainedTimeToPass=None
        #ref info
        self.trainRef=None
    def reset(self):
        self.tauLeft=0
        self.tauRight=0
        self.remainedTimeToPass=None
        #ref info
        self.trainRef=None
   
class Section:
    def __init__(self,sectionIndex,name,length,station1,station2):
        self.sectionIndex=sectionIndex
        self.name=name
        self.tracks=[SectionTrack(0,length)]
        self.station1=station1
        self.station2=station2
    def reset(self):
        self.tracks[0].reset()

    def NormalizationObvEncodeing(self,maxVelocity,thTimeValue):
        """
        """
        tauLeft=self.tracks[0].tauLeft
        tauLeft=min(tauLeft,thTimeValue)
        trainPriority=0
        trainVelocity=0
        trainStatu=0
        remainedTimeToPass=0
        reverse=None
        maxRemainderTime=None
        if isinstance(self.tracks[0].trainRef,RailTrain):
            trainPriority=self.tracks[0].trainRef.priority
            trainVelocity=self.tracks[0].trainRef.velocity
            trainStatu=self.tracks[0].trainRef.statu.value
            maxRemainderTime=math.ceil(self.tracks[0].secLength/trainVelocity)
            remainedTimeToPass=self.tracks[0].remainedTimeToPass
            
            assert self.tracks[0].trainRef.NextSource()['stationObjRef'] is not None,f'train on a section but its next resource is None'
            if self.tracks[0].trainRef.NextSource()['stationObjRef'] is self.station1:
                reverse=True
            else:
                reverse=False
        else:
            remainedTimeToPass=0
            maxRemainderTime=1
        tauRight=self.tracks[0].tauRight
        tauRight=min(tauRight,thTimeValue)
        if reverse is not None:
            if reverse:
                remainedTimeToPass=maxRemainderTime-remainedTimeToPass
        SectionCode = []
        SectionCode.extend(self.station1.NormalizationObvEncodeing(maxVelocity,thTimeValue))
        SectionCode.extend([tauLeft/thTimeValue,trainPriority/3,trainVelocity/maxVelocity,trainStatu/(len(list(RailTrainStatu))-1),remainedTimeToPass/maxRemainderTime,tauRight/thTimeValue])
        SectionCode.extend(self.station2.NormalizationObvEncodeing(maxVelocity,thTimeValue))
        assert max(SectionCode)<=1,f'exist value is bigger than 1'
        assert min(SectionCode)>=0,f'exist value is lower than 0'
        return SectionCode
        
    def GetTheSectionLeftTimer(self):
        return self.tracks[0].tauLeft
    def GetTheSectionRightTimer(self):
        return self.tracks[0].tauRight
    def SetTheSectionLeftTimer(self,newCnt):
        self.tracks[0].tauLeft=max(self.tracks[0].tauLeft,newCnt)
    def SetTheSectionRightTimer(self,newCnt):
        self.tracks[0].tauRight=max(self.tracks[0].tauRight,newCnt)
    def GetTheRemainedTimeToPass(self):
        return self.tracks[0].remainedTimeToPass
    def SetTheRemainedTimeToPass(self,newTime):
        self.tracks[0].remainedTimeToPass=newTime
    def ValidCheckingForEnteringFromLeftSide(self):
        if self.tracks[0].trainRef is not None:
            return False
        if self.tracks[0].tauLeft !=0:
            return False
        return True
    def ValidCheckingForEnteringFromRightSide(self):
        if self.tracks[0].trainRef is not None:
            return False
        if self.tracks[0].tauRight !=0:
            return False
        return True

class Lines:
    TAUA2A=3
    TAUA2D=3
    TAUD2A=3
    TAUS2S=3
    TAUMINDWELL=4
    TAUMAXDWELL=120
    TAUMAXSCHEDULINGTIME=1440
    def __init__(self,stationNames,stationTrackCntList,sectionIndexList,sectionLengths):
        self.stations=[]
        self.sections=[]
        self.initNxG=nx.Graph()
        
        for i in range(0,len(stationNames)):
            station=Station(i,stationNames[i],stationTrackCntList[i])
            self.stations.append(station)
            self.initNxG.add_node(i,label=f'index={i},maxc={stationTrackCntList[i]},name={stationNames[i]}',stationObjRef=station)#车站编号作为节点
        for i in range(0,len(sectionIndexList)):
            pair=sectionIndexList[i]
            length=sectionLengths[i]
            if pair[0]>pair[1]:
                tmp=pair[0]
                pair[0]=pair[1]
                pair[1]=tmp
            section=Section(f'{pair[0]}-{pair[1]}',f'section-{self.stations[pair[0]].name}-{self.stations[pair[1]].name}',length,self.stations[pair[0]],self.stations[pair[1]])
            self.sections.append(section)
            self.initNxG.add_edge(pair[0],pair[1],weight=length,label=f'index={pair[0]}-{pair[1]},name=section-{self.stations[pair[0]].name}-{self.stations[pair[1]].name},length={length}',sectionObjRef=section)
        keyIndex=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 13), (4, 14), (5, 6), (6, 7), (7, 8), (8, 9), (10, 11), (11, 12), (12, 13), (14, 15), (15, 16), (16, 17), (17, 18)]
        self.orderedSections=[]
        for secIndex in keyIndex:
            for section in self.sections:
                k,v=secIndex
                if section.sectionIndex==f'{k}-{v}' or section.sectionIndex==f'{v}-{k}':
                    self.orderedSections.append(section)
        self.sections=self.orderedSections

    def reset(self):
        """
        """
        for station in self.stations:
            for stationTrack in station.tracks:
                stationTrack.reset()
            station.reset()
        for section in self.sections:
            section.tracks[0].reset()
            section.reset()
        
    def Serialization(self):
        if self.initNxG is None:
            return[[],[],[]]
        else:
            nodesList=list(self.initNxG.nodes())
            adjMtr=nx.adjacency_matrix(self.initNxG,nodelist=nodesList).toarray().tolist()
            nodesCode=[[node,data['label'],data['stationObjRef'].Serialization()] for node,data in self.initNxG.nodes(data=True)]
            edgesCode=[[o,d,data['label'],data['sectionObjRef'].Serialization()] for o,d,data in self.initNxG.edges(data=True)]
            return[adjMtr,nodesCode,edgesCode]

    def SideOfABasedBOnGraph(self,sectionOrStationA,sectionOrStationB):
        """
        
        """
        assert (isinstance(sectionOrStationA,Section) and isinstance(sectionOrStationB,Station)) or (isinstance(sectionOrStationA,Station) and isinstance(sectionOrStationB,Section)),f"Type error:A is {type(sectionOrStationA)}, B is {type(sectionOrStationB)}"
        if isinstance(sectionOrStationA,Section) and isinstance(sectionOrStationB,Station):
            edgeKeys=[(k0,k1) for k0,k1,attris in self.initNxG.edges(data=True) if attris['sectionObjRef'] is sectionOrStationA]
            sectionOrStationKeyA=edgeKeys[0]
            nodeKeys=[key for key,attris in self.initNxG.nodes(data=True) if attris['stationObjRef'] is sectionOrStationB]           
            sectionOrStationKeyB=nodeKeys[0]
            if sectionOrStationKeyA[0]==sectionOrStationKeyB:
                return RelativeDirections.RightSide
            elif sectionOrStationKeyA[1]==sectionOrStationKeyB:
                return RelativeDirections.LeftSide
            else:
                assert 1==2,"A and B are not connected"
        else:
            nodeKeys=[key for key,attris in self.initNxG.nodes(data=True) if attris['stationObjRef'] is sectionOrStationA]
            edgeKeys=[(k0,k1) for k0,k1,attris in self.initNxG.edges(data=True) if attris['sectionObjRef'] is sectionOrStationB]
            sectionOrStationKeyA=nodeKeys[0]
            sectionOrStationKeyB=edgeKeys[0]
            if sectionOrStationKeyA==sectionOrStationKeyB[0]:
                return RelativeDirections.LeftSide
            elif sectionOrStationKeyA==sectionOrStationKeyB[1]:
                return RelativeDirections.RightSide
            else:
                assert 1==2,"A and B are not connected"
    
    def isTrainValidToMove(self,railTrain):
        assert railTrain.statu==RailTrainStatu.InOperation,f'the train is not InOperation statu'
        assert railTrain.currentNode is not None,f'train is not in a station'
        assert isinstance(railTrain.currentNode['stationObjRef'],Station),f'railTrain.currentNode[stationObjRef] is not Station'
        if railTrain.currentNode is railTrain.shortestPathInGraph[-1]:
            return True
        else:
            section=railTrain.shortestPathInGraph[railTrain.shortestPathInGraph.index(railTrain.currentNode)+1]['sectionObjRef']
            station=railTrain.currentNode['stationObjRef']
            stationTrack=railTrain.GetTheTrack()
            sectionIsOnTheSideofStation=self.SideOfABasedBOnGraph(section,station)
            if sectionIsOnTheSideofStation==RelativeDirections.LeftSide:
                if (stationTrack.GetTheStationTrackDwellTimer()>=Lines.TAUMINDWELL or stationTrack.GetTheStationTrackDwellTimer()==0) and section.GetTheSectionRightTimer()==0 and section.tracks[0].trainRef is None:
                    return True
                else:
                    return False
            else:
                if (stationTrack.GetTheStationTrackDwellTimer()>=Lines.TAUMINDWELL or stationTrack.GetTheStationTrackDwellTimer()==0) and section.GetTheSectionLeftTimer()==0 and section.tracks[0].trainRef is None:
                    return True
                else:
                    return False

class RailTrain:
    def __init__(self,index,initTime,oStationIndex,dStationIndex,path,velocity,priority):
        self.trainIndex=index
        self.initTime=initTime
        self.oStationIndex=oStationIndex
        self.dStationIndex=dStationIndex
        self.velocity=velocity
        self.priority=priority
        self.statu=RailTrainStatu.WaitStart
        self.path=path
        self.shortestPathInGraph=None
        self.currentNode=None
        self.currentEdge=None
        self.trRecord=[]
        self.actionToExecute=None
        self.isValidMove=None
        self.isMaxDwellTime=False
    
    def reset(self):
        self.statu=RailTrainStatu.WaitStart
        self.currentNode=None
        self.currentEdge=None
        self.trRecord=[]
        self.actionToExecute=None
        self.isValidMove=None
        self.isMaxDwellTime=False

    def Serialization(self):
        shortestCode=[]
        if self.shortestPathInGraph is None:
            pass
        else:
            shortestCode=[resource['label'] for resource in self.shortestPathInGraph]
        currentNodeCode=[]
        if self.currentNode is None:
            pass
        else:
            currentNodeCode=[self.currentNode['label']]
        currentEdgeCode=[]
        if self.currentEdge is None:
            pass
        else:
            currentEdgeCode=[self.currentEdge['label']]
        trRecordCode=[]
        if len(self.trRecord)==0:
            pass
        else:
            trRecordCode=[[tr[0],tr[1],tr[2],tr[3].name,tr[4].name] for tr in self.trRecord]
        return[self.trainIndex,self.initTime,self.oStationIndex,self.dStationIndex,self.velocity,self.priority,self.statu.name,self.path,shortestCode,currentNodeCode,currentEdgeCode,trRecordCode,self.actionToExecute.name]
    
    def GetTheTrack(self):
        if self.currentEdge is not None:
            if isinstance(self.currentEdge['sectionObjRef'],Section):
                return self.currentEdge['sectionObjRef'].tracks[0]
        if self.currentNode is not None:
            if isinstance(self.currentNode['stationObjRef'],Station):
                for track in self.currentNode['stationObjRef'].tracks:
                    if track.trainRef is self:
                        return track
            else:
                print('        SimSys:'+type(self.currentNode['stationObjRef']))
        assert self.statu!=RailTrainStatu.InOperation,f'the train {self.trainIndex} is InOperation, but cannot find the currentTrack'
        return None

    def NextSource(self):
        """
        
        """
        index=None
        if self.currentNode is not None:
            index=self.shortestPathInGraph.index(self.currentNode)+1
        else:
            index=self.shortestPathInGraph.index(self.currentEdge)+1
        assert index is not None,f'both currentNode and currentEdge are None, but the NextSource is called'
        if index>len(self.shortestPathInGraph)-1:
            return None
        else:
            return self.shortestPathInGraph[index]
       
class SimulatorBasedLinesAndTrains:
    InstanceParamsCopy=None
    InitTrains=None
    MinPD=1000000
    def __init__(self,FixedLineParams,LinePlanParams):
        stationNamesList=FixedLineParams['stationNamesList']
        stationTrackCntList=FixedLineParams['stationTrackCntList']
        sectionIndexList=FixedLineParams['sectionIndexList']
        sectionLengths=FixedLineParams['sectionLengths']

        initTimeList=LinePlanParams['initTimeList']
        oStaIndexList=LinePlanParams['oStaIndexList']
        dStaIndexList=LinePlanParams['dStaIndexList']
        velocityList=LinePlanParams['velocityList']
        priorityList=LinePlanParams['priorityList']
        if SimulatorBasedLinesAndTrains.InstanceParamsCopy is None:
            SimulatorBasedLinesAndTrains.InstanceParamsCopy=copy.deepcopy((stationNamesList,stationTrackCntList,sectionIndexList,sectionLengths,initTimeList,oStaIndexList,dStaIndexList,velocityList,priorityList))
       
        self.lines=Lines(stationNames=stationNamesList,stationTrackCntList=stationTrackCntList,sectionIndexList=sectionIndexList,sectionLengths=sectionLengths)
       
        self.railTrains=[]
        
        for i in range(0,len(initTimeList)):
           
            oStationIndex=oStaIndexList[i]
            dStationIndex=dStaIndexList[i]
            shortestPath=nx.shortest_path(self.lines.initNxG,source=oStationIndex,target=dStationIndex)
            railTrain=RailTrain(i,initTime=initTimeList[i],oStationIndex=oStaIndexList[i],dStationIndex=dStaIndexList[i],path=shortestPath,velocity=velocityList[i],priority=priorityList[i])
            self.railTrains.append(railTrain)
            
            shortestResourcePathInGraph=[]
            nodePath=[self.lines.initNxG.nodes[s] for s in shortestPath]
            secPath=[self.lines.initNxG[shortestPath[s]][shortestPath[s+1]] for s in range(0,len(shortestPath)-1)]
            for s in range(0,len(nodePath)-1):
                shortestResourcePathInGraph.append(nodePath[s])
                shortestResourcePathInGraph.append(secPath[s])
            shortestResourcePathInGraph.append(nodePath[-1])
            self.railTrains[i].shortestPathInGraph=shortestResourcePathInGraph
          
        self.statu=SimulatorStatu.WaitStart
        self.tau=0
    
        self.heuristic=False
       
        self.pd=0
        
        self.dwellRewardBase=-0.1 
        self.failedBase=-10 
        self.invalidActionBase=-0.3
        self.trainReachNextStattin=1
        self.sectionRewardBaseline=0
        self.stationRewardBaseline=0
        self.trainReachDestStation=3
        self.totalSuccessReward=5 
    
        self.sectionRewardDicts={section:self.sectionRewardBaseline for section in self.lines.sections}
        self.stationRewardDicts={station:self.stationRewardBaseline for station in self.lines.stations}
      
        self.sectionDoneDicts={section:False for section in self.lines.sections}
        if SimulatorBasedLinesAndTrains.InitTrains is None:
            SimulatorBasedLinesAndTrains.InitTrains=self.railTrains

    def resetTheRewards(self):
        self.sectionRewardDicts={section:self.sectionRewardBaseline for section in self.lines.sections}
        self.stationRewardDicts={station:self.stationRewardBaseline for station in self.lines.stations}
        self.sectionDoneDicts={section:False for section in self.lines.sections}
          
    def CountStationDelta(self,station):
        """
        
        """
        trainCnt=0
        for track in station.tracks:
            if track.trainRef is None:
                trainCnt+=1
        station.delta= len(station.tracks)-trainCnt
    
    def Reset(self):
        """
        
        """
       
        self.statu=SimulatorStatu.WaitStart
        self.tau=0
        self.heuristic=False
        self.pd=0
      
        self.sectionRewardDicts={section:self.sectionRewardBaseline for section in self.lines.sections}
        self.stationRewardDicts={station:self.stationRewardBaseline for station in self.lines.stations}
 
        self.sectionDoneDicts={section:False for section in self.lines.sections}

        self.lines.reset()

        for railTrain in self.railTrains:
            railTrain.reset()
            
    def Start(self):
        """
       
        """
        assert self.statu==SimulatorStatu.CompletedWithFailure or self.statu==SimulatorStatu.CompletedWithSuccess or self.statu==SimulatorStatu.WaitStart,f'system staut error, can not reStart the simulator'
        assert isinstance(self.lines,Lines),f'lines obj is None'
        assert len(self.railTrains)>0,f'railTrains is None'
    
        self.statu=SimulatorStatu.InOperation
        actionBool,lastTrackRef, lastStationRef, lastStationNodeRef, lastTrainRef=self.AppendNewTrainEvent()
        if actionBool == False:
            print('Terminal Step:Instance initialization Fail')
            print('Simulator exits')
            return False,True,self.State()
        return True,False,self.State()
    
    def State(self):
        """
      
        """
        lines=self.lines
        for train in self.railTrains:
            if train.statu==RailTrainStatu.InOperation:
                if train.currentNode is not None:
                    train.isValidMove=lines.isTrainValidToMove(train)
                else:
                    train.isValidMove=None
            else:
                train.isValidMove=None

        return self.lines
    
    def ReceiveAndExecuteDecisions(self,orderedDecisionTupList):
        """
       
        """
        dictTmp=dict(orderedDecisionTupList)
        orderedDecisionTupList=list(dictTmp.items())
        assert len(orderedDecisionTupList)==len([train for train in self.railTrains if train.statu==RailTrainStatu.InOperation]),f'the num of railTrains ({len([train for train in self.railTrains if train.statu==RailTrainStatu.InOperation])}) is not equal to the num of decisions ({len(orderedDecisionTupList)})'
        responses=[]
        for (railTrain,SchedulingActiontup) in orderedDecisionTupList:
            response=self.TrainExecuteSchedulingAction(railTrain,SchedulingActiontup)
            responses.append((railTrain,SchedulingActiontup,response))
            if response == ResponseToTheSchedulingAction.ValidActionWithFailExecuting:
                self.statu=SimulatorStatu.CompletedWithFailure
            if response==ResponseToTheSchedulingAction.InvalidActionWithoutExecuting:
                sectionOrStation=None
                trainCurrentResource=railTrain.currentNode
                if trainCurrentResource is None:
                    trainCurrentResource=railTrain.currentEdge
                    sectionOrStation=trainCurrentResource['sectionObjRef']
                else:
                    sectionOrStation=trainCurrentResource['stationObjRef']
                if isinstance(sectionOrStation,Station):
                    self.stationRewardDicts[sectionOrStation]+=self.invalidActionBase*railTrain.priority
                else:
                    self.sectionRewardDicts[sectionOrStation]+=self.invalidActionBase*railTrain.priority
            if response==ResponseToTheSchedulingAction.ValidActionWithFailExecuting:
                sectionOrStation=None
                trainCurrentResource=railTrain.currentNode
                if trainCurrentResource is None:
                    trainCurrentResource=railTrain.currentEdge
                    sectionOrStation=trainCurrentResource['sectionObjRef']
                else:
                    sectionOrStation=trainCurrentResource['stationObjRef']
                if isinstance(sectionOrStation,Station):
                    if railTrain.isMaxDwellTime:
                        self.stationRewardDicts[sectionOrStation]+=self.invalidActionBase*railTrain.priority*10 
                        railTrain.isMaxDwellTime=False
                    self.stationRewardDicts[sectionOrStation]+=self.failedBase*railTrain.priority
                   
                    nextSec=railTrain.NextSource()
                    nextSec=nextSec['sectionObjRef']
                    self.sectionDoneDicts[nextSec]=True
                else:
                    self.sectionRewardDicts[sectionOrStation]+=self.failedBase*railTrain.priority####RewardPointNo.7:FailedReward
                    
                    self.sectionDoneDicts[sectionOrStation]=True
        return responses

    def AppendNewTrainEvent(self):
        """
      
        """
        trainsNeedAppend=[]
        for train in self.railTrains:
            if train.statu==RailTrainStatu.WaitStart and train.initTime==self.tau:
                trainsNeedAppend.append(train)
        if len(trainsNeedAppend)==0:
            return True,None,None,None,None

        for train in trainsNeedAppend:
            train.statu=RailTrainStatu.InOperation
            assert len(train.shortestPathInGraph)>0,f'trainIndex:{train.trainIndex} Error: null path'
            stationNode=train.shortestPathInGraph[0]
            station=stationNode["stationObjRef"]
            track=station.FindValidTracksFromLeftside()
            fromLR='L'
            if track is None:
                track=station.FindValidTracksFromRightside()
                fromLR='R'
            if track is None:
                return False,None,station,stationNode,train
            assert track is not None,""
            isInitTrainOk,theTrack,theStation=(False,None,None)
            if fromLR=='L':
                isInitTrainOk,theTrack,theStation=station.StartTrainFromLeftside(train,self.tau)
            else:
                isInitTrainOk,theTrack,theStation=station.StartTrainFromRightside(train,self.tau)
            if (isInitTrainOk == False):
                return False,theTrack,theStation,stationNode,train
            
        return True,None,None,None,None

    def TrainExecuteSchedulingAction(self,railTrain,sa):
        """
      
        """
        if railTrain.statu==RailTrainStatu.WaitStart:
            if sa==SchedulingAction.MaintainTheStatu:
                return ResponseToTheSchedulingAction.ValidActionWithSuccessExecuting
            else:
                return ResponseToTheSchedulingAction.InvalidActionWithoutExecuting
        if railTrain.statu==RailTrainStatu.CompletedWithSuccess:
            if sa==SchedulingAction.MaintainTheStatu:
                return ResponseToTheSchedulingAction.ValidActionWithSuccessExecuting
            else:
                return ResponseToTheSchedulingAction.InvalidActionWithoutExecuting
        if railTrain.statu==RailTrainStatu.CompletedWithFailure:
            assert 1==2,f'the train {railTrain.trainIndex} has failed but get a scheduling action'
       
        if railTrain.statu==RailTrainStatu.InOperation and railTrain.currentEdge is not None and (sa!=SchedulingAction.MaintainTheStatu):
            assert isinstance(railTrain.currentEdge['sectionObjRef'] , Section),f'railTrain.currentEdge[sectionObjRef] is not a Section'
            return ResponseToTheSchedulingAction.InvalidActionWithoutExecuting
        if railTrain.statu==RailTrainStatu.InOperation and railTrain.currentEdge is not None and sa==SchedulingAction.MaintainTheStatu:
            assert isinstance(railTrain.currentEdge['sectionObjRef'] , Section),f'railTrain.currentEdge[sectionObjRef] is not a Section'
            return ResponseToTheSchedulingAction.ValidActionWithSuccessExecuting
     
        realRes=None
        if railTrain.statu==RailTrainStatu.InOperation and railTrain.currentNode is not None and (sa!=SchedulingAction.DwellAction and sa!=SchedulingAction.MoveAction):
            assert isinstance(railTrain.currentNode['stationObjRef'] , Station),f'railTrain.currentEdge[stationObjRef] is not a Station'
            print(f'        SimSys: the action {sa} has been changed to action SchedulingAction.DwellAction')
            sa=SchedulingAction.DwellAction
            realRes= ResponseToTheSchedulingAction.InvalidActionWithoutExecuting

        if self.heuristic:
            if not self.lines.isTrainValidToMove(railTrain=railTrain):
                sa=SchedulingAction.DwellAction
       
        if sa==SchedulingAction.MoveAction:
            
            if railTrain.currentNode is railTrain.shortestPathInGraph[-1]:
                self.TrainArrivalDestinationEvent(railTrain)
                return ResponseToTheSchedulingAction.ValidActionWithSuccessExecuting
            else:
                
                stationIndex=railTrain.shortestPathInGraph.index(railTrain.currentNode)
                currentStationNode=railTrain.shortestPathInGraph[stationIndex]
                currentStation=currentStationNode['stationObjRef']
                currentStationTrack=None
                for track in currentStation.tracks:
                    if track.trainRef is railTrain:
                        currentStationTrack=track
                assert isinstance(currentStationTrack , StationTrack),'Type error, expected a StationTrack object ref'
                nextSectionIndex=stationIndex+1
                nextSectionEdge=railTrain.shortestPathInGraph[nextSectionIndex]
                nextSection=nextSectionEdge['sectionObjRef']
                assert isinstance(nextSection , Section),'Type error, expected a Section object ref'
        
                EnterSectionFromLOrR=self.lines.SideOfABasedBOnGraph(currentStation,nextSection)
       
                couldMove=True
           
                if EnterSectionFromLOrR==RelativeDirections.LeftSide:
                    couldMove=couldMove and nextSection.ValidCheckingForEnteringFromLeftSide()
                    if couldMove==False:
                        print(f"    Reason: train {railTrain.trainIndex} at station {currentStation.name} enters invalid section {nextSection.name}")
                else:
                    couldMove=couldMove and nextSection.ValidCheckingForEnteringFromRightSide()
                    if couldMove==False:
                        print(f"    Reason: train {railTrain.trainIndex} at station {currentStation.name} enters invalid section {nextSection.name}")
                
    
                if currentStationTrack.GetTheStationTrackDwellTimer()==0:
    
                    pass
                else:
         
                    if currentStationTrack.GetTheStationTrackDwellTimer()<self.lines.TAUMINDWELL:#允许taumin--partc version
                        couldMove=couldMove and False
                        if couldMove==False:
                            print(f"    Reason: train {railTrain.trainIndex} at station {currentStation.name} enters section {nextSection.name} but not satisfies the TAUMINDWELL") 
                if couldMove == False:
                    railTrain.statu=RailTrainStatu.CompletedWithFailure
                    return ResponseToTheSchedulingAction.ValidActionWithFailExecuting
              
                railTrain.statu=RailTrainStatu.InOperation
                railTrain.currentNode=None
                railTrain.currentEdge=nextSectionEdge
                railTrain.trRecord.append((self.tau,nextSection.sectionIndex,nextSection.tracks[0].sIndex,SchedulingAction.MoveAction,railTrain.statu))
                railTrain.actionToExecute=SchedulingAction.MoveAction
             
                currentStationTrack.trainRef=None
                currentStationTrack.tauDwell=0
            
                nextSection.tracks[0].trainRef=railTrain
        
                DepartStationFromLOrR=None
                for direction in RelativeDirections:
                    if direction!=EnterSectionFromLOrR:
                        DepartStationFromLOrR=direction
                if DepartStationFromLOrR==RelativeDirections.RightSide:
                    currentStation.SetTheStationRightTimer(self.lines.TAUD2A)
                    for stationTrack in currentStation.tracks:
                        stationTrack.SetTheStationTrackRightTimer(self.lines.TAUD2A)
                    currentStationTrack.SetTheStationTrackLeftTimer(self.lines.TAUD2A)

                    nextSection.SetTheSectionLeftTimer(self.lines.TAUS2S+math.ceil(nextSection.tracks[0].secLength/railTrain.velocity))
                    nextSection.SetTheSectionRightTimer(self.lines.TAUA2D+math.ceil(nextSection.tracks[0].secLength/railTrain.velocity))
                    nextSection.SetTheRemainedTimeToPass(math.ceil(nextSection.tracks[0].secLength/railTrain.velocity))
                else:
                    currentStation.SetTheStationLeftTimer(self.lines.TAUD2A)
                    for stationTrack in currentStation.tracks:
                        stationTrack.SetTheStationTrackLeftTimer(self.lines.TAUD2A)
                    currentStationTrack.SetTheStationTrackRightTimer(self.lines.TAUD2A)

                    nextSection.SetTheSectionRightTimer(self.lines.TAUS2S+math.ceil(nextSection.tracks[0].secLength/railTrain.velocity))
                    nextSection.SetTheSectionLeftTimer(self.lines.TAUA2D+math.ceil(nextSection.tracks[0].secLength/railTrain.velocity))
                    nextSection.SetTheRemainedTimeToPass(math.ceil(nextSection.tracks[0].secLength/railTrain.velocity))
                return ResponseToTheSchedulingAction.ValidActionWithSuccessExecuting
        else:
            assert sa==SchedulingAction.DwellAction,f'sa is not SchedulingAction.DwellAction'
            stationIndex=railTrain.shortestPathInGraph.index(railTrain.currentNode)
            currentStationNode=railTrain.shortestPathInGraph[stationIndex]
            currentStation=currentStationNode['stationObjRef']
            currentStationTrack=None
            for track in currentStation.tracks:
                if track.trainRef is railTrain:
                    currentStationTrack=track
            assert isinstance(currentStationTrack , StationTrack),'Type error, expected a StationTrack object ref'
            nextSectionIndex=stationIndex+1
            nextSectionEdge=railTrain.shortestPathInGraph[nextSectionIndex]
            nextSection=nextSectionEdge['sectionObjRef']
            assert isinstance(nextSection , Section),'Type error, expected a Section object ref'

      
            railTrain.statu=RailTrainStatu.InOperation
            railTrain.currentNode=currentStationNode
            railTrain.currentEdge=None
            railTrain.trRecord.append((self.tau,currentStation.stationIndex,currentStationTrack.trackIndex,SchedulingAction.DwellAction,railTrain.statu))
            railTrain.actionToExecute=SchedulingAction.DwellAction
          
            currentStationTrack.trainRef=railTrain
             
            if currentStationTrack.GetTheStationTrackDwellTimer()>=self.lines.TAUMAXDWELL:
                railTrain.statu=RailTrainStatu.CompletedWithFailure
                railTrain.isMaxDwellTime=True
                print(f'    Reason: train {railTrain.trainIndex} reachs the max Dwell time')
                return ResponseToTheSchedulingAction.ValidActionWithFailExecuting#停站超时
            if realRes is None:
                return ResponseToTheSchedulingAction.ValidActionWithSuccessExecuting
            else:
                return realRes
    
    def TrainAutomaticallyArrivalStation(self,railTrain,tau):
        """
       
        """
        assert railTrain.statu==RailTrainStatu.InOperation,f'train statu error: expected a InOperation statu, but {railTrain.statu}'
        assert railTrain.currentNode is None,f'train currentNode is not None before it arrival'
        assert railTrain.currentEdge is not None,f'type error train currentEdge is not Section'
        assert isinstance(railTrain.currentEdge['sectionObjRef'] , Section),f'type error train currentEdge is not Section'
        currentSectionEdge=railTrain.currentEdge
        currentSectionEdgeIndexInPath=railTrain.shortestPathInGraph.index(currentSectionEdge)
        currentSection=currentSectionEdge['sectionObjRef']
        assert isinstance(currentSection , Section),f'type error: expected a Section but {type(currentSection)}'
        currentSectionTrack=currentSection.tracks[0]
        assert currentSectionTrack.remainedTimeToPass==0,f'event error: expected currentSectionTrack.remainedTimeToPass==0, but its {currentSectionTrack.remainedTimeToPass}'
        nextStationNode=railTrain.shortestPathInGraph[currentSectionEdgeIndexInPath+1]
        assert nextStationNode is not None,f'type error: expected a Station but {type(nextStationNode)}'
        assert isinstance(nextStationNode['stationObjRef'] , Station),f'type error: expected a Station but {type(nextStationNode)}'
        nextStation=nextStationNode['stationObjRef']

        stationIsOnTheSideOfSection=self.lines.SideOfABasedBOnGraph(nextStation,currentSection)
        if stationIsOnTheSideOfSection==RelativeDirections.RightSide:
            validTrack=nextStation.FindValidTracksFromLeftside()
            if validTrack is None:
                railTrain.statu=RailTrainStatu.CompletedWithFailure
                print(f'    Reason: train {railTrain.trainIndex} automatically arrives the station {nextStation.name} failed')
                return False,tau,railTrain,nextStation      
            
            railTrain.statu=RailTrainStatu.InOperation
            railTrain.currentEdge=None
            railTrain.currentNode=nextStationNode
            railTrain.trRecord.append((tau,currentSection.sectionIndex,currentSection.tracks[0].sIndex,SchedulingAction.AutoArrival,RailTrainStatu.InOperation))
            railTrain.actionToExecute=SchedulingAction.NeedDecision
            railTrain.trRecord.append((tau,nextStation.stationIndex,validTrack.trackIndex,SchedulingAction.NeedDecision,RailTrainStatu.InOperation))
      
            currentSectionTrack.remainedTimeToPass=None
            currentSectionTrack.trainRef=None
              
            validTrack.tauDwell=0
            validTrack.trainRef=railTrain
           
            currentSection.SetTheSectionRightTimer(self.lines.TAUA2D)
            currentSection.SetTheSectionLeftTimer(self.lines.TAUS2S)
              
            validTrack.SetTheStationTrackDwellTimer(0)
            validTrack.SetTheStationTrackLeftTimer(self.lines.TAUA2A)
            validTrack.SetTheStationTrackRightTimer(self.lines.TAUA2A)
     
            nextStation.SetTheStationLeftTimer(self.lines.TAUA2A)
            nextStation.SetTheStationRightTimer(self.lines.TAUA2A)
            
            self.sectionRewardDicts[currentSection]+=self.trainReachNextStattin
            return True,tau,railTrain,nextStation
        else:
            validTrack=nextStation.FindValidTracksFromRightside()
            if validTrack is None:
                railTrain.statu=RailTrainStatu.CompletedWithFailure
                print(f'    Reason: train {railTrain.trainIndex} automatically arrives the station {nextStation.name} failed')
                return False,tau,railTrain,nextStation
         
            railTrain.statu=RailTrainStatu.InOperation
            railTrain.currentEdge=None
            railTrain.currentNode=nextStationNode
            railTrain.trRecord.append((tau,currentSection.sectionIndex,currentSection.tracks[0].sIndex,SchedulingAction.AutoArrival,RailTrainStatu.InOperation))
            railTrain.actionToExecute=SchedulingAction.NeedDecision
            railTrain.trRecord.append((tau,nextStation.stationIndex,validTrack.trackIndex,SchedulingAction.NeedDecision,RailTrainStatu.InOperation))
         
            currentSectionTrack.remainedTimeToPass=None
            currentSectionTrack.trainRef=None
               
            validTrack.tauDwell=0
            validTrack.trainRef=railTrain
           
            currentSection.SetTheSectionLeftTimer(self.lines.TAUA2D)
            currentSection.SetTheSectionRightTimer(self.lines.TAUS2S)
                
            validTrack.SetTheStationTrackDwellTimer(0)
            validTrack.SetTheStationTrackRightTimer(self.lines.TAUA2A)
            validTrack.SetTheStationTrackLeftTimer(self.lines.TAUA2A)
           
            nextStation.SetTheStationRightTimer(self.lines.TAUA2A)
            nextStation.SetTheStationLeftTimer(self.lines.TAUA2A)
           
            self.sectionRewardDicts[currentSection]+=self.trainReachNextStattin 
            return True,tau,railTrain,nextStation
                        
    def TrainArrivalDestinationEvent(self,railTrain):
        assert railTrain.statu==RailTrainStatu.InOperation,f'train statu error: expected railTrain.statu==RailTrainStatu.InOperation, but {railTrain.statu}'
        assert railTrain.currentEdge is None,f'train at wrong position, expected at a station but the sectionNode is not None'
        assert railTrain.currentNode is not None,'train is not at a station'
        assert isinstance(railTrain.currentNode['stationObjRef'] , Station),f'train is not at a station'
        currentStation=railTrain.currentNode['stationObjRef']
        assert currentStation is railTrain.shortestPathInGraph[-1]['stationObjRef'],f'train is not at the destination station obj ref'
        stationTrack=None
        for statrack in currentStation.tracks:
            if statrack.trainRef is railTrain:
                stationTrack=statrack
        assert isinstance(stationTrack , StationTrack),'station track is None'
  
        railTrain.statu=RailTrainStatu.CompletedWithSuccess
        railTrain.currentNode=None
        railTrain.currentEdge=None
        railTrain.actionToExecute=None
        railTrain.trRecord.append((self.tau,currentStation.stationIndex,stationTrack.trackIndex,SchedulingAction.MoveAction,railTrain.statu))
    
        stationTrack.SetTheStationTrackDwellTimer(0)
        stationTrack.trainRef=None
        
        pastSection=railTrain.shortestPathInGraph[-2]['sectionObjRef']
        sectionIsOnTheSide=self.lines.SideOfABasedBOnGraph(pastSection,railTrain.shortestPathInGraph[-1]['stationObjRef'])
        assert isinstance(sectionIsOnTheSide , RelativeDirections),f'No side relation found'
        if sectionIsOnTheSide==RelativeDirections.LeftSide:
            stationTrack.SetTheStationTrackLeftTimer(self.lines.TAUD2A)
        else:
            stationTrack.SetTheStationTrackRightTimer(self.lines.TAUD2A)
     
        if sectionIsOnTheSide==RelativeDirections.LeftSide:
            currentStation.SetTheStationRightTimer(self.lines.TAUD2A)
        else:
            currentStation.SetTheStationLeftTimer(self.lines.TAUD2A)
    
        pathIndexes= [record[1] for record in railTrain.trRecord]
        for stationObj,r in self.stationRewardDicts.items():
            if stationObj.stationIndex in pathIndexes:
                self.stationRewardDicts[stationObj]+=self.trainReachDestStation
        for sectionObj,r in self.sectionRewardDicts.items():
            if sectionObj.sectionIndex in pathIndexes:
                self.sectionRewardDicts[sectionObj]+=self.trainReachDestStation


        self.stationRewardDicts[currentStation]+=self.trainReachDestStation ####RewardPointNo.12:TrainReachDestStation,+=self.trainReachDestStation
        print(f'     Env: train{railTrain.trainIndex} reached dest')
        return True,railTrain,currentStation,self.tau
    
    def IsDoneAndSimStatu(self):
        """
        return isdone,issuccess,isOutofTime
        """
        assert self.statu==SimulatorStatu.InOperation or self.statu==SimulatorStatu.CompletedWithFailure or self.statu==SimulatorStatu.CompletedWithSuccess,f'Simulator is in a  unavaliable statu'
        if self.statu==SimulatorStatu.CompletedWithSuccess:

            return True,True,False
        if self.statu==SimulatorStatu.CompletedWithFailure:
            return True,False,False
        assert self.statu==SimulatorStatu.InOperation,f'simulator statu error, statu:{self.statu}'
        if len(self.railTrains)==0:
            self.statu=SimulatorStatu.CompletedWithSuccess
            return True,True,False
        if self.tau>self.lines.TAUMAXSCHEDULINGTIME-2:
            self.statu=SimulatorStatu.CompletedWithFailure
            for station,reward in self.stationRewardDicts.items():
                self.stationRewardDicts[station]+=self.failedBase*10 
            for section,reward in self.sectionRewardDicts.items():
                self.sectionRewardDicts[section]+=self.failedBase*10
         
            for section in self.lines.sections:    
                self.sectionDoneDicts[section]=True
            return True,False,True
        isDone=False
        isSuccess=True
        for train in self.railTrains:
            if train.statu==RailTrainStatu.WaitStart or train.statu==RailTrainStatu.InOperation or train.statu==RailTrainStatu.CompletedWithFailure:
                isSuccess=isSuccess and False
            if train.statu==RailTrainStatu.CompletedWithFailure:
                self.statu=SimulatorStatu.CompletedWithFailure
                isDone=isDone or True
        isAllFinishedWithSuccess=True
        for train in self.railTrains:
            if train.statu!=RailTrainStatu.CompletedWithSuccess:
                isAllFinishedWithSuccess=isAllFinishedWithSuccess and False
        if isAllFinishedWithSuccess:
            self.statu=SimulatorStatu.CompletedWithSuccess
            isDone=True 
       
            for train in self.railTrains:
            
                for tr in train.trRecord:
                    if tr[3]==SchedulingAction.DwellAction:
                        self.pd+=train.priority
            if self.pd<SimulatorBasedLinesAndTrains.MinPD and self.pd>0:
                SimulatorBasedLinesAndTrains.MinPD=self.pd
            for section,reward in self.sectionRewardDicts.items():
                self.sectionRewardDicts[section]+=self.totalSuccessReward  
            for station,reward in self.stationRewardDicts.items():
                self.stationRewardDicts[station]+=self.totalSuccessReward 
       
            for section in self.lines.sections:    
                self.sectionDoneDicts[section]=True
        return isDone,isSuccess,False

    def TimeStepUpdate(self):
        """
    
        """
        #System
        assert self.statu==SimulatorStatu.InOperation,f'simulator statu error, expected SimulatorStatu.InOperation statu, but{SimulatorStatu.InOperation}'
        self.tau=self.tau+1
           
        for station in self.lines.stations:
            station.tauLeft=CountDown_0(station.tauLeft)
            station.tauRight=CountDown_0(station.tauRight)
                        
            for stationTrack in station.tracks:
                stationTrack.tauLeft=CountDown_0(stationTrack.tauLeft)
                stationTrack.tauRight=CountDown_0(stationTrack.tauRight)
                            
                if stationTrack.trainRef is None:
                    stationTrack.tauDwell=0
                else:
                    stationTrack.tauDwell=stationTrack.tauDwell+1
                    self.stationRewardDicts[station]+=self.dwellRewardBase*stationTrack.trainRef.priority
                    
        for section in self.lines.sections:
                        
            section.tracks[0].tauLeft=CountDown_0(section.tracks[0].tauLeft)
            section.tracks[0].tauRight=CountDown_0(section.tracks[0].tauRight)
            
            if section.tracks[0].remainedTimeToPass is None:
                pass
            else:
                assert section.tracks[0].remainedTimeToPass>0,f'the remainedTimer is still countdown but its value is zero, the section id:{section.sectionIndex},name:{section.name}'
                section.tracks[0].remainedTimeToPass=CountDown_0(section.tracks[0].remainedTimeToPass)
                assert isinstance(section.tracks[0].trainRef , RailTrain),f'the remainedTimer is still countdown but there is no railTrain in the section id:{section.sectionIndex},name:{section.name}'
        
        for railTrain in self.railTrains:
            if railTrain.statu==RailTrainStatu.InOperation and railTrain.currentNode is None and railTrain.currentEdge is not None:
                assert isinstance(railTrain.currentEdge['sectionObjRef'] , Section),f'None error or type error'
                currentSectionEdge=railTrain.currentEdge
                currentSection=currentSectionEdge['sectionObjRef']
                currentSectionTrack=currentSection.tracks[0]
                if currentSectionTrack.remainedTimeToPass==0:
                    isSuccessToAutoEnterStation,tau,theTrain,theStation=self.TrainAutomaticallyArrivalStation(railTrain=railTrain,tau=self.tau)
                    if isSuccessToAutoEnterStation == False:
                        
                        self.statu=SimulatorStatu.CompletedWithFailure
                        self.sectionRewardDicts[currentSection]+=self.failedBase*railTrain.priority####RewardPointNo.2:FailedReward
                        isTimeStepSuccess=False
                        print(f'    Reason: train {railTrain.trainIndex} enters the station {theStation.name} fail')
                        
                        self.sectionDoneDicts[currentSection]=True
                        return False
        
        isAppendSuccess,theTrack,theStation,theStationNode,thetrain =self.AppendNewTrainEvent()
        if not isAppendSuccess:
            self.stationRewardDicts[theStation]+=self.failedBase*thetrain.priority
            print(f'    Reason: train {railTrain.trainIndex} load to station {theStation.name} fail')
            for sec in self.lines.sections:
                sta1=sec.station1
                sta2=sec.station2
                if sta1 is theStation:
                    self.sectionDoneDicts[sec]=True
                elif sta2 is theStation:
                    self.sectionDoneDicts[sec]=True
                else:
                    pass 
            return False

        return True
   
    def FCFS_Greedy_MA_Heuristic_Policy(self,state):
        self.heuristic=True
        serviceTupleList=[(train,train.GetTheTrack().GetTheStationTrackDwellTimer()) for train in self.railTrains if train.currentNode is not None]
        orderedServiceTupleList=sorted(serviceTupleList,key=lambda x:x[1],reverse=True)
        orderedSchedulingActionTupleList=[]
        for i in range(0,len(orderedServiceTupleList)):
            train=orderedServiceTupleList[i][0]
            if train.statu==RailTrainStatu.InOperation and train.currentNode is not None:
                assert isinstance(train.currentNode['stationObjRef'] , Station),f'train.currentNode[stationObjRef] is not Station'
                
                if self.lines.isTrainValidToMove(train) == True:
                    orderedSchedulingActionTupleList.append((train,SchedulingAction.MoveAction))
                else:
                    orderedSchedulingActionTupleList.append((train,SchedulingAction.DwellAction))
            else:
                orderedSchedulingActionTupleList.append((train,SchedulingAction.MaintainTheStatu))

        for train in self.railTrains:
            if train in set([rt for (rt,sa) in orderedSchedulingActionTupleList]):
                pass
            else:
                orderedSchedulingActionTupleList.append((train,SchedulingAction.MaintainTheStatu))
        return orderedSchedulingActionTupleList

    def UpdateTrainIsValidMove(self):
        for train in self.railTrains:
            if train.statu==RailTrainStatu.InOperation:
                if train.currentNode is not None:
                    train.isValidMove=self.lines.isTrainValidToMove(train)
                else:
                    train.isValidMove=False
            else:
                train.isValidMove=False


class SimulatorStatu(Enum):
    WaitStart=0
    InOperation=1
    CompletedWithSuccess=2
    CompletedWithFailure=3

class RailTrainStatu(Enum):
    WaitStart=0
    InOperation=1
    CompletedWithSuccess=2
    CompletedWithFailure=3

class RelativeDirections(Enum):
    LeftSide=0
    RightSide=1

class SchedulingAction(Enum):
    """

    """
    InitAction=0
    DwellAction=1
    MoveAction=2
    AutoArrival=3
    #FinishAction=auto()
    NeedDecision=4
    MaintainTheStatu=5

class ResponseToTheSchedulingAction(Enum):
    """

    """
    InvalidActionWithoutExecuting=0
    ValidActionWithSuccessExecuting=1
    ValidActionWithFailExecuting=2

def CountDown_0(tau):
    assert isinstance(tau,int),f'parameter tau is not a int num'
    assert tau>=0,f'parameter tau is less than 0'
    if tau==0:
        return 0
    else:
        return tau-1  
     