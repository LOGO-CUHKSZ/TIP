import argparse
import os
import torch
import numpy as np
import pickle


def generate_cycles(Nsamples,d, min_cycle = 3, **kwargs):

    labels = np.random.randint(2,size = Nsamples)
    x_list = []
    edge_list = []
    for n in range(Nsamples):
        Nnodes = np.random.randint(10,20)
        if labels[n]:

            edge_index = torch.stack((torch.arange(Nnodes),(1+torch.arange(Nnodes))%Nnodes))
            edge_index = torch.cat((edge_index, torch.tensor([int(0), int(Nnodes/2)]).view(2, 1)), dim=1)
            
            x = torch.randn(Nnodes,d)
        else:
            edge_index = torch.stack((torch.arange(Nnodes),(1+torch.arange(Nnodes))%Nnodes))
            x = torch.randn(Nnodes,d)
        
        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),1)
        x_list += [x]
        edge_list += [edge_index]
            

    os.makedirs(f"./Cycles_{min_cycle}/", exist_ok=True)
            
    with open(f"./Cycles_{min_cycle}/graphs.txt", "wb") as fp:
        pickle.dump([x_list, edge_list], fp)
    torch.save(torch.tensor(labels),f"./Cycles_{min_cycle}/labels.pt")


def generate_2cycles(Nsamples,d, min_cycle = 3, **kwargs):

    labels = np.random.randint(2,size = Nsamples)
    x_list = []
    edge_list = []
    for n in range(Nsamples):
        Nnodes = np.random.randint(10,20)
        half_Nnodes = int(Nnodes / 2)
        if labels[n]:

            edge_index1 = torch.stack((torch.arange(half_Nnodes), (1 + torch.arange(half_Nnodes)) % half_Nnodes))
            edge_index2 = torch.stack((torch.arange(half_Nnodes) + half_Nnodes, (1 + torch.arange(half_Nnodes)) % half_Nnodes + half_Nnodes))
            connecting_edge = torch.tensor([[half_Nnodes - 1], [half_Nnodes]])
            edge_index = torch.cat((edge_index1, edge_index2, connecting_edge), dim=1)
            
            x = torch.randn(Nnodes,d)
        else:
            edge_index1 = torch.stack((torch.arange(half_Nnodes), (1 + torch.arange(half_Nnodes)) % half_Nnodes))
            edge_index2 = torch.stack((torch.arange(half_Nnodes) + half_Nnodes, (1 + torch.arange(half_Nnodes)) % half_Nnodes + half_Nnodes))
            edge_index = torch.cat((edge_index1, edge_index2), dim=1)

            x = torch.randn(Nnodes,d)
        
        edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(0,))),1)
        x_list += [x]
        edge_list += [edge_index]
            

    os.makedirs(f"./2Cycles_{min_cycle}/", exist_ok=True)
            
    with open(f"./2Cycles_{min_cycle}/graphs.txt", "wb") as fp:
        pickle.dump([x_list, edge_list], fp)
    torch.save(torch.tensor(labels),f"./2Cycles_{min_cycle}/labels.pt")
    

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Generation of Synthetic Graph Datasets')
    
    parser.add_argument('--Nsamples',type=int,default = 1000)
    parser.add_argument('--d',type=int,default = 3, help="Number of dimensions of the node features")
    parser.add_argument('--min_cycle',type=int, default = 3, help = "Size of smallest cycle in the Cycles graph")

    args = parser.parse_args()
    
    generate_cycles(args.Nsamples, args.d, args.min_cycle)

    print(f"You just generated {args.Nsamples} Graphs.")
