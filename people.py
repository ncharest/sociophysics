# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:19:24 2019

@author: Nate
"""
#%%
### Import Section
import numpy as np
import matplotlib.pyplot as plt
import random as ran
import copy
import networkx as nx

### Definitions
class agent:
    def __init__(self, opinion = [1,1], presentation = np.identity(2), aggression = 0.0, impression = 0.1, removal_threshold = 0.0):
        ### Initialize Variables
        self.opinion = np.asarray(opinion/np.linalg.norm(np.asarray(opinion)))
        self.presentation = presentation
        self.aggression = aggression
        self.impression = impression
        self.impression_innate = impression
        self.dimensionality = len(opinion)
        self.opinion_history = []
        self.opinion_history_pertime = []
        self.removal_threshold = removal_threshold
        self.end_relationship_flag = 0
        self.neighbor_history = []
        
    def interact(self, other):
        self.end_relationship_flag = 0
        ### We first calculated the interaction operator of the Self and the Other. The interaction operator can be written as:
        ### K_interaction = K_cross*K_self*K_other
        ### K_self is the presentation of Self
        ### K_other is the presentation of Other
        ### Both of these terms are based on the individual, and capture their tendency to moderate or extremize their own opinions.
        ### This could be adapted to reflect the Presented Opinion used in Buechel et al.
        ### K_cross is the interaction due to factors unique to the pairing of Self and Other. For simplicity we allow this to equal identity for this model.
        ### K_interaction must be unitary, as we will calculate the Commonality factor as < Self | K_interaction | Other >
        
        ### The interaction elements need not necessarily commute! We assume the Other is approaching the Self during an iteration, and thus the presentation of Other
        ### should affect the interaction before Self's!
        self.opinion_history.append(copy.deepcopy(self.opinion)[0])
        
        self.interaction = np.matmul(np.matmul(self.presentation, other.presentation),np.identity(self.dimensionality))
        self.commonality = np.matmul(self.opinion, np.matmul(self.interaction,other.opinion))
        if self.commonality < self.aggression:
#            print('Tribal Interaction Occurred')
            self.opinion = self.opinion - self.impression*np.matmul(self.interaction, other.opinion)
            self.opinion = self.opinion/np.linalg.norm(self.opinion)
            if ran.random() < self.removal_threshold:
                self.end_relationship_flag = 1
            else: 
                self.end_relationship_flag = 0
                
                
            
        
        else:
#            print('Amicable Interaction Occurred')
            self.opinion = self.opinion + self.impression*np.matmul(self.interaction, other.opinion)
            self.opinion = self.opinion/np.linalg.norm(self.opinion)
        
        for i in range(len(self.opinion)):
            if self.opinion[i] < 0.0:
#                print('Individual cannot be further radicalized')
                self.opinion[i] = 0.0
            self.opinion = self.opinion/np.linalg.norm(self.opinion)
       
        return self.end_relationship_flag
        
    
    def update_neighborhood_initial(self, neighbors):
        self.impression = self.impression_innate
        self.neighbors = neighbors
        self.num_neighbors = len(neighbors)
        if self.num_neighbors == 0:
            pass
        else:
            self.impression = self.impression / self.num_neighbors

        
    def update_neighborhood(self, neighbors):
        self.neighbors = neighbors
        self.num_neighbors = len(neighbors)


    
    def vote(self):
        self.probabilities = []
        for i in self.opinion:
            self.probabilities.append(i**2)
        self.probabilities = np.asarray(self.probabilities)
        self.boundries = [0.0]
        for i in self.probabilities:
            self.boundries.append(self.boundries[-1]+i)
        self.boundries = np.asarray(self.boundries)

        self.collapse_seed = ran.random()
        self.collapse_state = np.zeros(self.dimensionality)
        for i in range(len(self.boundries)-1):
            if self.collapse_seed < self.boundries[i+1] and self.collapse_seed >= self.boundries[i]:
                self.collapse_state[i] = 1
        return self.collapse_state
        
class social_network:
    def __init__(self, size=10, connectivity=3, seed=121791, agent_spec={'opinion' : [ran.uniform(0.0,1.0),ran.uniform(0.0,1.0)], 'aggression' : np.random.normal(0.7, 0.1), 'impression' : 0.1, 'removal_threshold' : np.random.normal(0.001, 0.0001)}):
        self.size = size
        self.connectivity = connectivity
        self.seed = seed
        ### Barabasi Albert graph is thought to be semblant of social networks online
        self.network = nx.barabasi_albert_graph(size, connectivity, seed)
        self.agents = {}
        self.flag = 0
        for i in range(size):
            ### READ READ READ
            ### I changed how the agents are generated and now they are all identical. This is obviously bad. Undo it! Undo the dictionary!
            ### THIS COMMENT HAS BEEN ADDRESSED. I need to think of a better way of implementing this.
            self.agents.update({i : agent(opinion= [ran.uniform(0.0,1.0),ran.uniform(0.0,1.0)], aggression = abs(np.random.normal(0.6, 0.2)), impression = abs(np.random.normal(0.05, 0.05)), removal_threshold = abs(np.random.normal(0.001, 0.00025)))})
        self.colors = []
        for i in range(size):
            self.colors.append(self.agents[i].opinion[0])
            
            
    def update_neighborhoods_initial(self):
        for i in range(self.size):
            self.agents[i].update_neighborhood_initial([n for n in self.network.neighbors(i)])
            
    def update_neighborhoods(self):
        for i in range(self.size):
            self.agents[i].update_neighborhood([n for n in self.network.neighbors(i)])
            
    def update_colors(self):
        self.colors = []
        for i in range(self.size):
            self.colors.append(self.agents[i].opinion[0])
            
    def advance_time(self):
        self.update_neighborhoods()
        for i in range(self.size):
            self.agents[i].opinion_history_pertime.append(self.agents[i].opinion[0])
            self.agents[i].neighbor_history.append(copy.deepcopy(self.agents[i].neighbors))

            for j in self.agents[i].neighbors:
#                print(self.agents[i].neighbors)
#                print(str(i)+' '+str(j)+' interacting...')
                if i == j:
                    pass
                else:
                    
                    self.flag = self.agents[i].interact(self.agents[j])
                    if self.flag == 1:
#                        print(str(i)+' '+str(j)+' Relationship Ended')
                        self.network.remove_edge(i, j)
                        self.update_neighborhoods_initial()
                    else:
                        pass
        self.flag = 0
        
            
        self.update_colors()
        
    def draw_network(self, Labels = 'True'):
        nx.draw(self.network, with_labels=Labels, node_color=self.colors, cmap = plt.cm.Blues)
        plt.show()
        print(min(self.colors))
        print(max(self.colors))
        
    def find_extremists(self):
        self.extremists = []
        for i in range(self.size):
            if self.agents[i].opinion[0] >= 0.90 or self.agents[i].opinion[0] <= 0.1:
                self.extremists.append((i, self.agents[i].opinion))
        return len(self.extremists)
    
    def connectedness(self):
        self.connected = []
        for i in range(self.size):
            self.connected.append((len(self.agents[i].neighbors), i))
        self.connected.sort(key=lambda x: x[0])
        
                
        
    
        
path = 'D:/projects/project-0_skunk/model_output/sociophysics/7/'
size = 100
SN = social_network(size = size, connectivity = 3)
SN0 = copy.deepcopy(SN)
SN.update_neighborhoods_initial()

plot_element = []

SN.draw_network()

for i in range(0,10000):
    SN.advance_time()

    if i%100 == 0:
        plt.subplot(1, 2, 1)
        for j in range(0,size):
            plt.plot(SN.agents[j].opinion_history)
        plt.title('Position By Interaction')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Opinion of Position A')
        plt.subplot(1, 2, 2)
        for k in range(0,size):
            plt.plot(SN.agents[k].opinion_history_pertime)
        plt.title('Position By Time')
        plt.xlabel('Number of Iterations')
        
        
        plt.savefig(path+'timestamp_'+str(i)) 
        plt.show()

identities = []

for i in range(size):
    identities.append([i,SN.agents[i].opinion[0],SN.agents[i].aggression, SN.agents[i].impression, len(SN.agents[i].neighbors)])
    
for i in range(size):
    print('Writing History of Agent '+str(i))
    fout = open(path+'/agents/'+'agent_'+str(i)+'.txt','w')
    fout.write("# ID # Opinion of Position 0 # Aggression # Impressionability # Neighbors at End # \n")
    fout.write(str(identities[i])+"\n")
    for j in range(len(SN.agents[i].opinion_history_pertime)):    
        line = str(SN.agents[i].opinion_history_pertime[j])+" "+str(SN.agents[i].neighbor_history[j])+"\n"
        fout.write(line)
       
   
    fout.close()

    

   
SN.draw_network()


    

    
#nx.draw(SN.network, with_labels='True', node_color=SN.colors, cmap = plt.cm.Blues)