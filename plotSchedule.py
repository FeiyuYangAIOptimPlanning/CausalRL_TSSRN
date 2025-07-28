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
from SimulatorClasses import SimulatorBasedLinesAndTrains,RailTrain,Lines,Section,Station,SimulatorStatu,RailTrainStatu,SchedulingAction,ResponseToTheSchedulingAction
import numpy as np
import matplotlib.pyplot as plt
import pickle

class trainLine:
    def __init__(self,train):
        self.train=train
        self.trainIdx=train.trainIndex
        self.trainPriority=train.priority
        self.path=train.path
        self.stationIOTime=[]
        self.stationIdxs=[]
        self.stationTrackIdxs=[]
        self.sectionIOTime=[]
        self.sectionIdxs=[]
        self.getXYs()
    def getXYs(self):
       
        for t,objIdx,trackIdx,_,_ in self.train.trRecord:
            if isinstance(objIdx,int):
                self.stationIOTime.append(t)
                self.stationIdxs.append(objIdx)
                self.stationTrackIdxs.append(trackIdx)
       
        for t,objIdx,trackIdx,_,_ in self.train.trRecord:
            if isinstance(objIdx,str):
                write_t=None
                if t in self.stationIOTime:
                    write_t=t-1
                else:
                    write_t=t
                self.sectionIOTime.append(write_t)
                self.sectionIdxs.append(objIdx)

class ScheduleFig:
    def __init__(self,stationIdx,stationNames,stationMeters):
        self.stationIdx=stationIdx
        self.ymeters=stationMeters
        self.ynames=stationNames

        self.yTickMode=stationMeters
        self.yLabelMode=stationMeters
        self.figObj=plt.figure()
        self.ax=self.figObj.add_subplot(111)
        
    def setYmode(self,mode):
        if mode=='meter':
            self.yTickMode=self.ymeters
            self.yLabelMode=[str(m) for m in self.ymeters]
        if mode=='index':
            self.yTickMode=self.stationIdx
            self.yLabelMode=[str(i) for i in self.stationIdx]
        if mode=='name':
            self.yTickMode=self.ymeters
            self.yLabelMode=self.ynames
    
        self.ax.set_yticks(self.yTickMode)
        self.ax.set_yticklabels(self.yLabelMode)

    def plot(self,ts,stationIdxs,linelabel='',trainPriority=0):
        yindexs=self.find_indexes_in_list(stationIdxs,self.stationIdx)
        meters=[self.ymeters[i] for i in yindexs]
        if self.yTickMode==self.ymeters:
            self.ax.plot(ts,meters,label=linelabel,color=self.setColor(trainPriority))
        else:
            self.ax.plot(ts,stationIdxs,label=linelabel,color=self.setColor(trainPriority))
    
    def setMarker(self,trainPriority):
        if trainPriority==1:
            return 'v'
        if trainPriority==2:
            return 's'
        if trainPriority==3:
            return '^'
        return 'x'
    
    def setColor(self,trainPriority):
        if trainPriority==1:
            return 'g'
        if trainPriority==2:
            return 'k'
        if trainPriority==3:
            return 'r'
        return 'b'

    def show(self):
    
        self.ax.set_yticks(self.yTickMode)
        self.ax.set_yticklabels(self.yLabelMode)
       
        self.ax.grid(axis='y', linestyle='--')
        self.ax.xaxis.grid(True, color='g', linestyle='--')
        self.figObj.show()

    def find_indexes_in_list(self,elements, reference_list):
        return [reference_list.index(element) for element in elements]
    
class SchedulePloter:
    def __init__(self,simlinesObj):
        self.Simlines=simlinesObj
        self.SimGraph=simlinesObj.lines.initNxG
        self.SimOrderedSections=simlinesObj.lines.orderedSections
        self.SimStations=simlinesObj.lines.stations
        
        self.trainLines=[]
        for train in simlinesObj.railTrains:
            self.trainLines.append(trainLine(train=train))
        paths=[]
        for trainline in self.trainLines:
            paths.append(trainline.path)
        maxTime=max([max(tl.stationIOTime) for tl in self.trainLines])
        merged_paths=self.merge_paths(self.SimGraph,paths)
        self.figs=[]
        for path in merged_paths:
            self.figs.append(ScheduleFig(path,[f'station {stationIdx}' for stationIdx in path],self.calculate_path_length(self.SimGraph,path)))
        for fig in self.figs:
            fig.setYmode('name')
            fig.ax.set_xticks(range(0,int(maxTime)+30,30))

        for trainline in self.trainLines:
            isPloted=False
            for schedulefig in self.figs:
                if self.is_subpath(trainline.path,schedulefig.stationIdx):
                    schedulefig.plot(trainline.stationIOTime,trainline.stationIdxs,f'train {trainline.trainIdx}',trainline.trainPriority)
                    isPloted=True
                    break
            if isPloted==False:
                print('Schedule plot error')
        for fig in self.figs:
            fig.show()
    def merge_paths(self,G, paths):

        merged_paths = []
        for path in paths:
      
            if any(set(path).issubset(set(merged_path)) for merged_path in merged_paths):
                continue
  
            for merged_path in merged_paths:
                if set(merged_path).issubset(set(path)):
                    merged_paths.remove(merged_path)
            merged_paths.append(path)
        return merged_paths
    
    def calculate_path_length(self,G, path):
        lengths = [0] 
        for i in range(1, len(path)):
 
            length = lengths[-1] + G[path[i-1]][path[i]]['weight']
            lengths.append(length)
        return lengths

    def appendPlotLine(self,trainline):
        if isinstance(trainline,trainLine):
            self.plotLines.append(trainline)

    def is_subpath(self, path1, path2):
       
        if (all(node in path2 for node in path1) and
            all(path2.index(path1[i]) < path2.index(path1[i + 1]) for i in range(len(path1) - 1))) or \
        (all(node in path2 for node in reversed(path1)) and
            all(path2.index(path1[i]) > path2.index(path1[i + 1]) for i in range(len(path1) - 1))):
            return True
        return False


simulator1=None
with open('success_simulator_2024-03-30-02-11_pd_1123.pkl','rb') as file:
    simulator1=pickle.load(file)
scheduleplot=SchedulePloter(simulator1)
print('')


