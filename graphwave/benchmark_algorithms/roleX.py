# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:57:52 2017

@author: Lab41: Github: Circulo/circulo/algorithms/rolx.py
####  https://github.com/Lab41/Circulo/blob/master/circulo/algorithms/rolx.py


Set of functions to compute the RolX featurization
"""

import sys
import math
import igraph
import numpy as np
from numpy.linalg import lstsq
from numpy import dot
from scipy.cluster.vq import kmeans2, vq
from scipy.linalg import norm
from scipy.optimize import minimize
from sklearn.decomposition import NMF
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import torch
sys.path.append('../')
from utils.utils import read_real_datasets, NodeClassificationDataset, MLP, DataSplit

def extract_rolx_roles(G, roles=5):
    """
    Top-level function. Extracts node-role matrix and sensemaking role-feature matrix as necessary.
    """
    print("Creating Vertex Features matrix")
    V = vertex_features(G)
    #print("V is a %s by %s matrix." % V.shape)

    basis, coef = get_factorization(V, roles)

    H = basis
    #print("Node-role matrix is of dimensions %s by %s" % H.shape)
    #print(H)

    K = make_sense(G, H)
    #print("Role-feature matrix is of dimensions %s by %s" % K.shape)
    #print(K)

    return H, K

def extract_rolx_roles_bis(G,V, roles=2):
    """
        Top-level function. Extracts node-role matrix and sensemaking role-feature matrix as necessary.
        Inputs a matrux
        """
    
    basis, coef = get_factorization(V, roles)
    
    H = basis
    print("Node-role matrix is of dimensions %s by %s" % H.shape)
    #print(H)
    
    K = make_sense(G, H)
    print("Role-feature matrix is of dimensions %s by %s" % K.shape)
    #print(K)
    
    return H, K

def recursive_feature(G, f, n):
    """
        G: iGraph graph with annotations
        func: string containing function name
        n: int, recursion level
        Computes the given function recursively on each vertex
        Current precondition: already have run the computation for G, func, n-1.
    """
    return np.matrix(recursive_feature_array(G,f,n))

def recursive_feature_array(G, func, n):
    """
    Computes recursive features of the graph G for the provided function of G, returning
    the matrix representing the nth level of the recursion.
    """
    attr_name = "_rolx_" + func.__name__ + "_" + str(n)

    if attr_name in G.vs.attributes():
        result = np.array(G.vs[attr_name])
        return result

    if n==0:
        stats = func(G)
        result = np.array([[x] for x in stats])
        result = result * 1.0 
        G.vs[attr_name] = result
        return result

    prev_stats = recursive_feature_array(G, func, n-1)
    all_neighbor_stats = []
    for v in G.vs:
        neighbors = G.neighbors(v)
        degree = len(neighbors)
        if degree == 0: 
            neighbor_avgs = neighbor_sums = np.zeros(prev_stats[0].size)
        else: 
            prev_neighbor_stats = [prev_stats[x] for x in neighbors]
            neighbor_sums_vec = sum(prev_neighbor_stats)
            neighbor_avgs_vec = neighbor_sums_vec / degree

        v_stats = np.concatenate((neighbor_sums_vec, neighbor_avgs_vec), axis=0)
        all_neighbor_stats.append(v_stats)

    G.vs[attr_name] = all_neighbor_stats
    return all_neighbor_stats

def approx_linear_solution(w, A, threshold=1e-15):
    '''
    Checks if w is linearly dependent on the columns of A, this is done by solving the least squares problem (LSP)
        min || w - Ax ||_2^2
         x
    and checking if || w - Ax_star || <= threshold, where x_star is the arg_minimizer of the LSP
    w: column vector
    A: matrix
    threshold: int
    '''
    x0 = np.zeros(A.shape[1])
    x_star, residuals, rank, s = lstsq(A, w)
    norm_residual = norm(residuals)
    result = True if norm_residual <= threshold else False
    return (result, norm_residual, x_star)

def degree(G):
    """ Auxiliary function to calculate the degree of each element of G. """
    return G.degree()

def vertex_egonet(G, v):
    """ Computes the number of edges in the ego network of the vertex v. """
    ego_network = G.induced_subgraph(G.neighborhood(v))
    ego_edges = ego_network.ecount()
    return ego_edges

def egonet(G):
    """ Computes the ego network for all vertices v in G. """
    return [vertex_egonet(G, v) for v in G.vs]

def vertex_egonet_out(G, v):
    """ Computes the outgoing edges from the ego network of the vertex v in G. """
    neighbors = G.neighborhood(v)
    ego_network = G.induced_subgraph(neighbors)
    ego_edges = ego_network.ecount()
    degree_sum = sum([G.degree(v) for v in neighbors])
    out_edges = degree_sum - 2*ego_edges #Summing over degree will doublecount every edge within the ego network
    return out_edges

def egonet_out(G):
    """ Computes the number of outgoing ego network edges for every vertex in G. """
    return [vertex_egonet_out(G, v) for v in G.vs]

def vertex_features(g):
    """ 
    Constructs a vertex feature matrix using recursive feature generation, then uses least-squares solving
    to eliminate those exhibiting approximate linear dependence.
    """
    G = g.copy()
    num_rows = G.vcount()

    features = [degree, egonet, egonet_out]
    V = np.matrix(np.zeros((num_rows, 16*len(features))))

    next_feature_col = 0
    for feature in features:
        base = recursive_feature(G, feature, 0)
        base = base/norm(base)
        V = add_col(V, base, next_feature_col)

        next_feature_col += 1
        level = 1

        accepted_features = True
        while accepted_features:
            accepted_features = False
            feature_matrix = recursive_feature(G, feature, level)
            rows, cols = feature_matrix.shape

            for i in range(cols):
                b = feature_matrix[:,i]
                b = b/norm(b)
                mat = V[:,:next_feature_col]
                threshold = 10.0**(-15+level)
                (is_approx_soln, _, _) = approx_linear_solution(b, mat, threshold)
                if not is_approx_soln:
                    V = add_col(V, b, next_feature_col)
                    next_feature_col += 1
                    accepted_features = True
            level += 1

    return V[:, :next_feature_col]

def add_col(V, b, insert_col):
    """ Add the given column b to the matrix V, enlarging the matrix if necessary. """ 
    rows, cols = V.shape
    if insert_col == cols: # need to resize V
        zeros = np.matrix(np.zeros((rows, 1)))
        V = np.concatenate((V, zeros), axis=1)
    V[:, insert_col] = b
    return V

def kmeans_quantize(M, bits):
    """ Performs k-means quantization on the given matrix. Returns the encoded matrix and the number of bits needed for encoding it. """
    k = 2**bits

    obs = np.asarray(M).reshape(-1)
    centroid, label = kmeans2(obs, k)

    enc_M = [centroid[v] for v in label]
    enc_M = np.matrix(enc_M).reshape(M.shape)

    return enc_M, (bits * enc_M.size)

def kl_divergence(A,B):
    """ Computes the Kullback-Leibler divergence of the two matrices A and B. """
    a = np.asarray(A, dtype=np.float)
    b = np.asarray(B, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def description_length(V, fctr_res, bits=10):
    """ Computes the length necessary to describe the given model with the given number of bits. """
    W = fctr_res[0]
    H = fctr_res[1]

    enc_W, enc_W_cost = kmeans_quantize(W, bits)
    enc_H, enc_H_cost = kmeans_quantize(H, bits)

    enc_cost = enc_W_cost + enc_H_cost
    err_cost = kl_divergence(V, enc_W*enc_H)

    return enc_W, enc_H, enc_cost, err_cost

def standardize_rows(M):
    """ Distribute the rows of the cost matrix normally to allow for accurate comparisons of error and description
    cost. """
    rv = np.matrix(M)
    for i in range(rv.shape[0]):
        mean = np.mean(M[i, :])
        stdev = np.std(M[i, :])
        rv[i, :]= (M[i, :]- mean)/stdev
    return rv

# def standardize(M):
#    m_flat = np.asarray(M).reshape(-1)
#    mean = np.mean(m_flat)
#    stdev = np.std(m_flat)
#    m_flat = (m_flat - mean)/stdev
#  
#    return m_flat.reshape(M.shape)

def get_factorization(V, num_roles):
    """ Obtains a nonnegative matrix factorization of the matrix V with num_roles intermediate roles. """
    model = NMF(n_components=num_roles, init='random', random_state=0)
    model.fit(V)
    
    node_roles = model.transform(V)
    role_features = model.components_
    
    return np.matrix(node_roles), np.matrix(role_features)

def get_optimal_factorization(V, min_roles=2, max_roles=6, min_bits=1, max_bits=10):
    """ Uses grid search to find the optimal parameter number and encoding of the given matrix factorization. """ 
    max_roles = min(max_roles, V.shape[1]) # Can't have more possible roles than features

    num_role_options = max_roles - min_roles
    num_bit_options = max_bits - min_bits

    mat_enc_cost = np.zeros((num_role_options, num_bit_options))
    mat_err_cost = np.zeros((num_role_options, num_bit_options))
    mat_fctr_res = [[0] * num_bit_options] * num_role_options

    # Setup and run the factorization problem
    for i in range(num_role_options):
        rank = min_roles + i
        fctr_res = get_factorization(V, rank)

        for j in range(num_bit_options):
            bits = min_bits + j
            enc_W, enc_H, enc_cost, err_cost = description_length(V, fctr_res, bits)
            mat_enc_cost[i,j] = enc_cost
            mat_err_cost[i,j] = err_cost
            mat_fctr_res[i][j] = (enc_W, enc_H) 

    mat_std_enc_cost = standardize_rows(mat_enc_cost)
    mat_std_err_cost = standardize_rows(mat_err_cost)

    mat_total_cost = mat_enc_cost + mat_err_cost
    mat_total_std_cost = mat_std_enc_cost + mat_std_err_cost

   # print mat_total_cost
    print('min cost @', idx, ' or at ', min_coord)

    print("rank, bits, enc_cost, err_cost, total_cost, std_enc_cost, std_err_cost, std_total_cost") 
    for i in range(num_role_options):
        for j in range(num_bit_options):
            rank = min_roles + i
            bits = min_bits + j

            enc_cost       = mat_enc_cost[i,j]
            err_cost       = mat_err_cost[i,j]

            std_enc_cost   = mat_std_enc_cost[i,j]
            std_err_cost   = mat_std_err_cost[i,j]

            total_cost     = mat_total_cost[i,j]
            total_std_cost = mat_total_std_cost[i,j]

            print("%s, %s, (%s, %s, %s), (%s, %s, %s)" % (rank, bits, 
                    enc_cost, err_cost, total_cost, std_enc_cost, std_err_cost, total_std_cost))

    min_idx = mat_total_std_cost.argmin()
    min_coord = np.unravel_index(min_idx, mat_total_std_cost.shape)
    min_role_index, min_bit_index = min_coord

    min_role_value = min_role_index + min_roles
    min_bit_value = min_bit_index + min_bits
    
    min_std_enc_cost = mat_std_enc_cost[min_coord]
    min_std_err_cost = mat_std_err_cost[min_coord]
    min_total_std_cost = mat_total_std_cost[min_coord]
    print("%s, %s, (%s, %s, %s)" % (min_role_value, min_bit_value, min_std_enc_cost, min_std_err_cost, min_total_std_cost))

    return mat_fctr_res[min_role_index][min_bit_index]

def make_sense(G, H):
    """ Given graph G and node-role matrix H, returns a role-feature matrix K for sensemaking analyses of roles. """
    features = [ 'betweenness', 'closeness', 'degree', 'diversity', 'eccentricity', 'pagerank', 'personalized_pagerank', 'strength' ]
    feature_fns = [ getattr(G, f) for f in features ]
    feature_matrix = [ func() for func in feature_fns ]
    feature_matrix = np.matrix(feature_matrix).transpose()

    #print(feature_matrix)

    M = feature_matrix
    for i in range(M.shape[1]):
        M[:,i] = M[:,i] / norm(M[:,i])

    K = complete_factor(H, M, h_on_left=True)
    #print(K)

    return K

def sense_residual_left_factor(W, H, M):
    W = np.matrix(W).reshape((M.shape[0], H.shape[0]))
    return norm(M - W*H)

def sense_residual_right_factor(K, H, M):
    K = np.matrix(K).reshape((H.shape[1], M.shape[1]))
   # print(M.shape,H.shape,K.shape)
    return norm(M - H*K)

def complete_factor(H, M, h_on_left=True):
    """Given nonnegative matrix M and a nonnegative factor H of M, finds the other (nonnegative) factor of M.
       H: known factor of matrix M.
       M: product matrix.
       h_on_left: boolean, true if H is the left factor of M, false if H is the right factor.
       If H is left factor, find the matrix K such that HK=M. If H is the right factor, finds W such that WH=M
       Result is an appropriately-sized matrix. """

    if h_on_left:
        shape = (H.shape[1], M.shape[1])
        residual = sense_residual_right_factor
    else:
        shape = (M.shape[0], H.shape[0])
        residual = sense_residual_left_factor

    size = shape[0] * shape[1]
    guess = np.random.rand(size)
    bounds = [(0, None)] * size # (all elements of matrix must be nonnegative)
    result = minimize(residual, guess, args=(H, M), method='L-BFGS-B', bounds=bounds)

    x = result["x"]
    G = np.matrix(x).reshape(shape)
    return G

def main(G_path, role_id_num):
    G = igraph.Graph.Read_GML(G_path)
    return extract_rolx_roles(G, roles=role_id_num)
    # if len(argv) > 0:
        # roles = role_id_num
        #A = nx.adjacency_matrix(G).todense()
        #Gi = igraph.Graph.Adjacency((A > 0).tolist())
        #test = extract_rolx_roles(Gi, roles=roles)
        ### Define a distance based on these distribution over roles
        # D_roleX = distance_nodes(test)
        #return extract_rolx_roles(G, roles=roles)
    # else:

    # return H, K

def read_roleid(path_to_file):
    role_id = []
    with open(path_to_file) as f:
        contents = f.readlines()
        for content in contents:
            role_id.append(int(content))
    return role_id

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
    plt.show()
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


if __name__ == "__main__":
   role_id = read_roleid("../cycle.roleid")
   role_id_num = len(set(role_id))
   embeddings, attributes = main(sys.argv[1:][0], role_id_num = role_id_num)
   labels_pred, colors, trans_data, nb_clust = cluster_graph(role_id, embeddings)
   draw_pca(role_id, embeddings)
   hom, comp, ami, nb_clust, ch, sil = unsupervised_evaluate(colors, labels_pred, trans_data, nb_clust)
   print('Homogeneity \t Completeness \t AMI \t nb clusters \t CH \t  Silhouette \n')
   print(hom, comp, ami, nb_clust, ch, sil)
   # Real world data
   G, labels = read_real_datasets("cornell")
   nx.write_gml(G, "cornell.gml")
   embeddings, attributes = main("cornell.gml", role_id_num=15)
   node_embeddings, node_labels = embeddings, labels
   input_dims = node_embeddings.shape
   class_number = int(max(node_labels)) + 1
   FNN = MLP(num_layers=5, input_dim=input_dims[1], hidden_dim=input_dims[1] // 2, output_dim=class_number)
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(FNN.parameters())
   dataset = NodeClassificationDataset(node_embeddings, node_labels)
   split = DataSplit(dataset, shuffle=True)
   train_loader, val_loader, test_loader = split.get_split(batch_size=64, num_workers=0)
   # train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
   for epoch in range(50):
       for i, data in enumerate(train_loader, 0):
           # data = data.to(device)
           inputs, labels = data
           y_pred = FNN(inputs.float())
           loss = criterion(y_pred, labels)
           print(epoch, i, loss.item())
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
       correct = 0
       total = 0
       with torch.no_grad():
           total_label_len = 0
           total_correct = 0
           for data in val_loader:
               inputs, labels = data
               total_label_len += len(labels)
               outputs = FNN(inputs.float())
               _, predicted = torch.max(outputs.data, 1)
               correct = torch.sum(predicted == labels)
               total_correct += correct.item()
           print(total_correct * 1.0 / total_label_len)
   correct = 0
   total = 0
   with torch.no_grad():
       for data in test_loader:
           inputs, labels = data
           outputs = FNN(inputs.float())
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()

   print('Accuracy of the network on test set: %d %%' % (
           100 * correct / total))
