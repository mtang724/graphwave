#### Tests like paper
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import seaborn as sb
import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import sys

sys.path.append('../')
import graphwave as gw
from shapes.shapes import *
from shapes.build_graph import *
# from distances.distances_signature import *
from characteristic_functions import *

def read_roleid(path_to_file):
    role_id_fl = []
    with open(path_to_file) as f:
        contents = f.readlines()
        for content in contents:
            role_id_fl.append(float(content))
    role_id = []
    for role in role_id_fl:
        role_id.append(int(role))
    return role_id

def graph_generator(width_basis=15, basis_type = "cycle", n_shapes = 5, shape_list=[[["house"]]], identifier = 'AA', add_edges = 0):
    ################################### EXAMPLE TO BUILD A SIMPLE REGULAR STRUCTURE ##########
    ## REGULAR STRUCTURE: the most simple structure:  basis + n small patterns of a single type
    ### 1. Choose the basis (cycle, torus or chain)
    ### 2. Add the shapes
    list_shapes = []
    for shape in shape_list:
        list_shapes += shape * n_shapes
    print(list_shapes)

    ### 3. Give a name to the graph
    name_graph = 'houses' + identifier
    sb.set_style('white')

    ### 4. Pass all these parameters to the Graph Structure
    G, communities, plugins, role_id = build_structure(width_basis, basis_type, list_shapes, start=0,
                                                                   add_random_edges=add_edges,
                                                                   plot=True, savefig=False)
    return G, role_id, identifier

# name_graph='barbell'
# sb.set_style('white')
# G , role_id = barbel_graph(0, 8, 5,plot=True)
# G, role_id, name_graph = graph_generator(identifier="cycle")
#
# nx.write_gml(G, "{}.gml".format(name_graph))
# nx.write_edgelist(G, "{}.edgelist".format(name_graph))

# with open("{}.roleid".format(name_graph), "w") as f:
#     for id in role_id:
#         f.write(str(id) + "\n")
#
# print ('nb of nodes in the graph: ', G.number_of_nodes())
# print ('nb of edges in the graph: ', G.number_of_edges())


def cluster_graph(role_id, node_embeddings):
    colors = role_id
    nb_clust = len(np.unique(role_id))
    pca = PCA(n_components=5)
    trans_data = pca.fit_transform(StandardScaler().fit_transform(node_embeddings))
    km = KMeans(n_clusters=nb_clust)
    km.fit(trans_data)
    labels_pred = km.labels_

    ######## Params for plotting
    cmapx = plt.get_cmap('rainbow')
    x = np.linspace(0, 1, nb_clust + 1)
    col = [cmapx(xx) for xx in x]
    markers = {0: '*', 1: '.', 2: ',', 3: 'o', 4: 'v', 5: '^', 6: '<', 7: '>', 8: 3, 9: 'd', 10: '+', 11: 'x',
               12: 'D', 13: '|', 14: '_', 15: 4, 16: 0, 17: 1, 18: 2, 19: 6, 20: 7}

    for c in np.unique(role_id):
        indc = [i for i, x in enumerate(role_id) if x == c]
        plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
                    c=np.array(col)[list(np.array(labels_pred)[indc])],
                    marker=markers[c % len(markers)], s=300)

    labels = role_id
    for label, c, x, y in zip(labels, labels_pred, trans_data[:, 0], trans_data[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    # plt.show()
    return labels_pred, colors, trans_data, nb_clust


def unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust):
    ami = sk.metrics.adjusted_mutual_info_score(colors, labels_pred)
    sil = sk.metrics.silhouette_score(trans_data, labels_pred, metric='euclidean')
    ch = sk.metrics.calinski_harabasz_score(trans_data, labels_pred)
    hom = sk.metrics.homogeneity_score(colors, labels_pred)
    comp = sk.metrics.completeness_score(colors, labels_pred)
    #print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
    #print(str(hom) + '\t' + str(comp) + '\t' + str(ami) + '\t' + str(nb_clust) + '\t' + str(ch) + '\t' + str(sil))
    return hom, comp, ami, nb_clust, ch, sil

def draw_pca(role_id, node_embeddings):
    cmap = plt.get_cmap('hot')
    x_range = np.linspace(0, 0.8, len(np.unique(role_id)))
    coloring = {u: cmap(x_range[i]) for i, u in enumerate(np.unique(role_id))}
    node_color = [coloring[role_id[i]] for i in range(len(role_id))]
    pca = PCA(n_components=2)
    node_embedded = StandardScaler().fit_transform(node_embeddings)
    principalComponents = pca.fit_transform(node_embedded)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2'])
    principalDf['target'] = role_id
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 PCA Components', fontsize=20)
    targets = np.unique(role_id)
    for target in zip(targets):
        color = coloring[target[0]]
        indicesToKeep = principalDf['target'] == target
        ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1'],
                   principalDf.loc[indicesToKeep, 'principal component 2'],
                   s=50,
                   c=color)
    ax.legend(targets)
    ax.grid()
    plt.show()

def average(lst):
    return sum(lst) / len(lst)

from graphwave import graphwave_alg

# homs = []
# comps = []
# amis = []
# chs = []
# sils = []
# for i in range(10):
#     G, role_id, name_graph = graph_generator(identifier="Varied{}".format(i), width_basis=6, shape_list=[[["house", 5]]], add_edges=0, n_shapes=2)
#
#     #nx.write_gml(G, "new_graphs/{}.gml".format(name_graph))
#     #nx.write_edgelist(G, "new_graphs/{}.edgelist".format(name_graph))
#
#     #with open("new_graphs/{}.roleid".format(name_graph), "w") as f:
#         # for id in role_id:
#         #     f.write(str(id) + "\n")
#
#     print('nb of nodes in the graph: ', G.number_of_nodes())
#     print('nb of edges in the graph: ', G.number_of_edges())
#     chi,heat_print, taus = graphwave_alg(G, np.linspace(0,100,25), taus=range(19,21), verbose=True)
#     mapping_inv={i: taus[i] for i in range(len(taus))}
#     mapping={float(v): k for k,v in mapping_inv.items()}
#
#     from sklearn.decomposition import PCA
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.cluster import KMeans
#
#     colors = role_id
#     nb_clust = len(np.unique(colors))
#     pca = PCA(n_components=5)
#
#     trans_data = pca.fit_transform(StandardScaler().fit_transform(chi))
#     km = sk.cluster.KMeans(n_clusters=nb_clust)
#     km.fit(trans_data)
#     labels_pred = km.labels_
#     ######## Params for plotting
#     cmapx = plt.get_cmap('rainbow')
#     x = np.linspace(0,1,np.max(labels_pred) + 1)
#     col = [cmapx(xx) for xx in x ]
#     markers = {0:'*',1: '.', 2:',',3: 'o',4: 'v',5: '^',6: '<',7: '>',8: 3 ,\
#                9:'d',10: '+',11:'x',12:'D',13: '|',14: '_',15:4,16:0,17:1,\
#                18:2,19:6,20:7}
#     ########
#
#     for c in np.unique(colors):
#             indc = [i for i, x in enumerate(colors) if x == c]
#             #print indc
#             plt.scatter(trans_data[indc, 0], trans_data[indc, 1],
#                         c=np.array(col)[list(np.array(labels_pred)[indc])],
#                         marker=markers[c%len(markers)], s=500)
#     labels = colors
#     for label,c, x, y in zip(labels,labels_pred, trans_data[:, 0], trans_data[:, 1]):
#                 plt.annotate(label,xy=(x, y), xytext=(0, 0), textcoords='offset points')
#
#     labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, chi)
#     draw_pca(role_id, chi)
#     hom, comp, ami, nb_clust, ch, sil = unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust)
#     homs.append(hom)
#     comps.append(comp)
#     amis.append(ami)
#     chs.append(ch)
#     sils.append(sil)
# print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
# print(average(homs), average(comps), average(amis), nb_clust, average(chs), average(sils))

from utils.utils import read_real_datasets, NodeClassificationDataset, DataSplit, MLP
import torch
import statistics

acc = []
for i in range(10):
    G, labels = read_real_datasets("wisconsin")
    G = nx.read_edgelist("realdatasets/film.edgelist")
    node_labels = read_roleid("realdatasets/np_film.txt")
    node_labels = torch.FloatTensor(node_labels)
    chi, heat_print, taus = graphwave_alg(G, np.linspace(0,10,100), taus=[1.0], verbose=True)
    node_embeddings, node_labels = chi, node_labels
    print(node_embeddings.shape, node_labels.shape)
    input_dims = node_embeddings.shape
    class_number = int(max(node_labels)) + 1
    FNN = MLP(num_layers=5, input_dim=input_dims[1], hidden_dim=input_dims[1]//2, output_dim=class_number)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(FNN.parameters())
    dataset = NodeClassificationDataset(node_embeddings, node_labels)
    split = DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
    # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    best = -float('inf')
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # data = data.to(device)
            inputs, labels = data
            inputs = inputs
            labels = labels
            y_pred = FNN(inputs.float())
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                correct = 0
                total = 0
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs
                    labels = labels
                    outputs = FNN(inputs.float())
                    _, predicted = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                    total += labels.size(0)
                    correct += torch.sum(predicted == labels)
            if correct / total > best:
                best = correct / total
                torch.save(FNN.state_dict(), 'best_mlp1.pkl')
            print(str(epoch), correct / total)
    with torch.no_grad():
        FNN.load_state_dict(torch.load('best_mlp1.pkl'))
        correct = 0
        total = 0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs
            labels = labels
            outputs = FNN(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += torch.sum(predicted == labels)
    print((correct / total).item())
    acc.append((correct / total).item())
print("mean:")
print(statistics.mean(acc))
print("std:")
print(statistics.stdev(acc))
