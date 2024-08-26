from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
from collections import deque
import networkx as nx
from tqdm import tqdm


def import_lastfm_asia_data(FILE_PATH):
    lastfm_edges_list = []
    with open(FILE_PATH, 'r') as data:
        for index, line in enumerate(data):
            if index == 0:
                continue
            node1, node2 = line.strip().split(',')
            lastfm_edges_list.append((int(node1), int(node2)))
    return lastfm_edges_list


def create_adj_matrix(edge_list):
    G = nx.Graph()

    # Add edges to the graph
    G.add_edges_from(edge_list)

    # Generate the adjacency matrix
    adj_matrix = nx.to_numpy_array(G, dtype=int)

    # Print the adjacency matrix
    # print(adj_matrix)
    return np.array(adj_matrix)


def import_wiki_vote_data(FILE_PATH):
    edge_list = []
    with open(FILE_PATH, 'r') as data:
        for line in data:
            if line.startswith("#"):
                continue
            edge_list.append(line.split())
    return edge_list


def make_nodes_sequential(edge_list):
    node_mapping = {}
    current_label = 0

    for src, dest in edge_list:
        if src not in node_mapping:
            node_mapping[src] = current_label
            current_label += 1
        if dest not in node_mapping:
            node_mapping[dest] = current_label
            current_label += 1
    return node_mapping


def create_adj_list(node_mapping, edge_list):
    sequential_adj_list = {}

    for src, dest in edge_list:
        new_src = node_mapping[src]
        new_dest = node_mapping[dest]

        if new_src not in sequential_adj_list:
            sequential_adj_list[new_src] = []
        if new_dest not in sequential_adj_list:
            sequential_adj_list[new_dest] = []

        sequential_adj_list[new_src].append(new_dest)
        sequential_adj_list[new_dest].append(new_src)
    return sequential_adj_list


def set_node_attributes(adj_list):
    node_attributes = {}
    for node in adj_list.keys():
        if node not in node_attributes:
            node_attributes[node] = {
                "color": "WHITE",
                "distance": float('inf'),
                "parent": [],
                "num_shortest_path": 0
            }
    return node_attributes


def plot_dendrogram(community_matrix):
    linkage_matrix = linkage(community_matrix, method="average")
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Dendrogram")
    plt.xlabel("Nodes")
    plt.ylabel("Distance")
    plt.show()


class GirvanNewman_:
    def BFS(self, graph_adj_list, graph_node_attributes, source_node):
        bfs_order = []
        graph_node_attributes[source_node]["color"] = "GREY"
        graph_node_attributes[source_node]["distance"] = 0
        graph_node_attributes[source_node]["num_shortest_path"] = 1

        Q = deque([source_node])
        while Q:
            current_node = Q.popleft()
            current_node_neighbours = graph_adj_list[current_node]
            bfs_order.append(current_node)

            for node in current_node_neighbours:
                if graph_node_attributes[node]["color"] == "WHITE":
                    graph_node_attributes[node]["color"] = "GREY"
                    graph_node_attributes[node]["distance"] = graph_node_attributes[current_node]["distance"] + 1
                    graph_node_attributes[node]["parent"].append(current_node)
                    Q.append(node)
                if graph_node_attributes[node]["distance"] == graph_node_attributes[current_node]["distance"] + 1:
                    graph_node_attributes[node]["num_shortest_path"] += graph_node_attributes[current_node]["num_shortest_path"]
                    if current_node not in graph_node_attributes[node]["parent"]:
                        graph_node_attributes[node]["parent"].append(
                            current_node)
            graph_node_attributes[current_node]["color"] = "BLACK"
        return graph_node_attributes, bfs_order

    def edge_centrality(self, node_attr, node_order, centrality_score_dict, node_value_dict):
        node_order = node_order[::-1]
        for index, node in enumerate(node_order):
            current_node_value = node_value_dict[node]
            current_node_shortest_path = node_attr[node]['num_shortest_path']
            current_node_parents = node_attr[node]['parent']
            for parent in current_node_parents:
                parent_shortest_path_num = node_attr[parent]['num_shortest_path']
                centrality_score = (
                    current_node_value) * (parent_shortest_path_num / current_node_shortest_path)
                node_value_dict[parent] += centrality_score
                edge = (parent, node)
                if edge not in centrality_score_dict:
                    centrality_score_dict[edge] = centrality_score
                else:
                    centrality_score_dict[edge] += centrality_score

    def create_node_value(self, adj_list):
        node_value = {}
        for index, node in enumerate(adj_list.keys()):
            if node not in node_value:
                node_value[node] = 1
        return node_value

    def remove_highest_centrality_edges(self, graph_adj_list, centrality_score):
        max_centrality = max(centrality_score.values())
        edges_to_remove = [
            edge for edge, score in centrality_score.items() if score == max_centrality]

        for edge in edges_to_remove:
            src, dest = edge
            graph_adj_list[src].remove(dest)
            graph_adj_list[dest].remove(src)
        return edges_to_remove

    def calculate_edge_betweenness(self, adj_list):
        centrality_score = {}
        for source_node in tqdm(adj_list.keys(), desc="Calculating edge betweenness"):
            node_value = self.create_node_value(adj_list)
            node_attr = set_node_attributes(adj_list)
            bfs_result, node_order = self.BFS(adj_list, node_attr, source_node)
            self.edge_centrality(bfs_result, node_order,
                                 centrality_score, node_value)
        return centrality_score

    def find_connected_components(self, graph_adj_list):
        visited = {node: False for node in graph_adj_list}
        components = []

        def dfs_iterative(start_node):
            stack = [start_node]
            component = []

            while stack:
                node = stack.pop()
                if not visited[node]:
                    visited[node] = True
                    component.append(node)
                    # Adding neighbors to the stack
                    for neighbor in graph_adj_list[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)

            return component

        for node in graph_adj_list:
            if not visited[node]:
                component = dfs_iterative(node)
                components.append(component)

        return components

    def girvan_newman_one_iter(self, graph_adj_list):
        inital_components = self.find_connected_components(graph_adj_list)
        num_nodes = len(graph_adj_list)
        communities_over_time = []
        node_indices = list(graph_adj_list.keys())
        count = 0
        while count == 0:
            count += 1
            centrality_score = self.calculate_edge_betweenness(graph_adj_list)
            if not centrality_score:
                break
            self.remove_highest_centrality_edges(
                graph_adj_list, centrality_score)

            # Find communities (connected components) at this iteration
            components = self.find_connected_components(graph_adj_list)
            community_labels = np.zeros(num_nodes, dtype=int) - 1

            for label, component in enumerate(components):
                for node in component:
                    community_labels[node_indices.index(node)] = label

            communities_over_time.append(community_labels)

            if len(components) == num_nodes:
                break

        communities_over_time_array = np.array(communities_over_time)
        return communities_over_time_array

    def girvan_newman_till_convergence(self, graph_adj_list):
        num_nodes = len(graph_adj_list)
        communities_over_time = []
        node_indices = list(graph_adj_list.keys())
        previous_num_communities = len(
            self.find_connected_components(graph_adj_list))
        initial_num_communities = previous_num_communities
        count = 1
        pbar = tqdm(total=15, desc="Iterations", leave=True)

        while True:
            centrality_score = self.calculate_edge_betweenness(graph_adj_list)
            if not centrality_score:
                break
            self.remove_highest_centrality_edges(
                graph_adj_list, centrality_score)

            # Find communities (connected components) at this iteration
            components = self.find_connected_components(graph_adj_list)
            current_num_communities = len(components)
            community_labels = np.zeros(num_nodes, dtype=int) - 1

            for label, component in enumerate(components):
                for node in component:
                    community_labels[node_indices.index(node)] = label

            # Add the current iteration's community labels to the list
            communities_over_time.append(community_labels.copy())

            # Check if the number of communities has increased
            if current_num_communities == initial_num_communities+2 or count == 15:
                break

            if previous_num_communities == current_num_communities:
                count += 1
            previous_num_communities = current_num_communities

            pbar.update(1)

        pbar.close()

        # Convert the list of community labels to a numpy array
        communities_over_time_array = np.array(communities_over_time)

        return communities_over_time_array


class Louvain:
    def compute_modularity(self, total_edges, adjacency_matrix, community):
        modularity = 0
        for node1 in community:
            for node2 in community:
                if adjacency_matrix[node1][node2] != 0:
                    node1_degree = sum(adjacency_matrix[node1])
                    node2_degree = sum(adjacency_matrix[node2])
                    modularity += adjacency_matrix[node1][node2] - \
                        (node1_degree * node2_degree) / (2 * total_edges)
        return modularity / (2 * total_edges)

    def louvain_phase1(self, adjacency_matrix, community_mapping, node_community, total_edges):
        num_nodes = len(adjacency_matrix)
        improvement = True

        while improvement:
            improvement = False
            modularity_before = self.calculate_total_modularity(
                adjacency_matrix, community_mapping, total_edges)

            for node in tqdm(range(num_nodes), desc="Louvain Phase 1", unit="node"):
                current_community = node_community[node]
                neighbors = np.where(adjacency_matrix[node] != 0)[0]
                max_modularity_change = -float('inf')
                max_modularity_change_community = None

                for neighbor in neighbors:
                    neighbor_community = node_community[neighbor]
                    if neighbor_community == current_community:
                        continue

                    # Calculate modularity before moving the node
                    modularity_before_move = self.compute_modularity(
                        total_edges, adjacency_matrix, community_mapping[current_community])

                    # Temporarily move node to neighbor's community
                    community_mapping[current_community].remove(node)
                    community_mapping[neighbor_community].append(node)
                    node_community[node] = neighbor_community

                    # Calculate modularity after moving the node
                    modularity_after_move = self.compute_modularity(
                        total_edges, adjacency_matrix, community_mapping[neighbor_community])

                    # Revert the move
                    community_mapping[neighbor_community].remove(node)
                    community_mapping[current_community].append(node)
                    node_community[node] = current_community

                    modularity_change = modularity_after_move - modularity_before_move
                    if modularity_change > max_modularity_change:
                        max_modularity_change = modularity_change
                        max_modularity_change_community = neighbor_community

                # Move the node to the community with the maximum modularity gain
                if max_modularity_change_community is not None:
                    community_mapping[current_community].remove(node)
                    community_mapping[max_modularity_change_community].append(
                        node)
                    node_community[node] = max_modularity_change_community
                    improvement = True

            modularity_after = self.calculate_total_modularity(
                adjacency_matrix, community_mapping, total_edges)
            if abs(modularity_after - modularity_before) < 1e-5:
                print("Modularity change below threshold. Stopping Phase 1.")
                break

        return community_mapping, node_community

    def calculate_total_modularity(self, adjacency_matrix, community_mapping, total_edges):
        total_modularity = 0
        for community in community_mapping.values():
            total_modularity += self.compute_modularity(
                total_edges, adjacency_matrix, community)
        return total_modularity

    def louvain_phase2(self, adjacency_matrix, community_mapping):
        new_num_nodes = len(community_mapping)
        new_adjacency_matrix = np.zeros((new_num_nodes, new_num_nodes))

        old_to_new_node = {}
        for new_node, (community, nodes) in enumerate(community_mapping.items()):
            for old_node in nodes:
                old_to_new_node[old_node] = new_node

        for old_node1 in range(len(adjacency_matrix)):
            for old_node2 in range(len(adjacency_matrix)):
                if old_node1 != old_node2:
                    new_node1 = old_to_new_node[old_node1]
                    new_node2 = old_to_new_node[old_node2]
                    if new_node1 != new_node2:
                        new_adjacency_matrix[new_node1][new_node2] += adjacency_matrix[old_node1][old_node2]

        new_community_array = np.zeros((len(adjacency_matrix), 1))
        for old_node in range(len(adjacency_matrix)):
            new_community_array[old_node] = old_to_new_node[old_node]

        return new_adjacency_matrix, new_community_array


def Girvan_Newman_one_level(edge_list):
    gn = GirvanNewman_()
    sequentail_node_mapping = make_nodes_sequential(edge_list)
    gn_adj_list = create_adj_list(sequentail_node_mapping, edge_list)
    graph_partition_list = gn.girvan_newman_one_iter(gn_adj_list)
    return graph_partition_list


def Girvan_Newman(edge_list):
    gn = GirvanNewman_()
    sequentail_node_mapping = make_nodes_sequential(edge_list)
    gn_adj_list = create_adj_list(sequentail_node_mapping, edge_list)
    graph_partition_list = gn.girvan_newman_till_convergence(gn_adj_list)
    return graph_partition_list


def visualise_dendogram(community_matrix):
    plot_dendrogram(community_matrix)


def louvain_one_iter(edge_list):
    total_edges = len(edge_list)
    adj_matrix = create_adj_matrix(edge_list)
    community_mapping = {}
    for index in range(len(adj_matrix)):
        community_mapping[index] = [index]

    lastfm_node_community = {}
    for index in range(len(adj_matrix)):
        lastfm_node_community[index] = index  # {node : community }

    louvain = Louvain()
    # for i in range(5):
    community_mapping, node_community = louvain.louvain_phase1(
        adj_matrix, community_mapping, lastfm_node_community, total_edges)
    adj_matrix, final_res = louvain.louvain_phase2(
        adj_matrix, community_mapping)
    return final_res


if __name__ == "__main__":

    ############ Answer qn 1-4 for wiki-vote data #################################################
    # Import wiki-vote.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row
    # is an edge connecting i<->j (entry in the first column is node i,
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_wiki = import_wiki_vote_data(
        "data/wiki-Vote.txt")

    # This is for question no. 1
    # graph_partition: graph_partitition is a nx1 numpy array where the rows corresponds to nodes in the network (0 to n-1) and
    #                  the elements of the array are the community ids of the corressponding nodes.
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    graph_partition_wiki = Girvan_Newman_one_level(
        nodes_connectivity_list_wiki)
    # print(graph_partition_wiki)
    # This is for question no. 2. Use the function
    # written for question no.1 iteratetively within this function.
    # community_mat is a n x m matrix, where m is the number of levels of Girvan-Newmann algorithm and n is the number of nodes in the network.
    # Columns of the matrix corresponds to the graph_partition which is a nx1 numpy array, as before, corresponding to each level of the algorithm.
    community_mat_wiki = Girvan_Newman(nodes_connectivity_list_wiki)

    # This is for question no. 3
    # Visualise dendogram for the communities obtained in question no. 2.
    # Save the dendogram as a .png file in the current directory.
    visualise_dendogram(community_mat_wiki)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before. Show the resulting communities after one iteration of the algorithm.
    graph_partition_louvain_wiki = louvain_one_iter(
        nodes_connectivity_list_wiki)
    # print(graph_partition_louvain_wiki)

    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import lastfm_asia_edges.csv
    nodes_connectivity_list_lastfm = import_lastfm_asia_data(
        "../data/lastfm_asia_edges.csv")

    # Question 1
    graph_partition_lastfm = Girvan_Newman_one_level(
        nodes_connectivity_list_lastfm)

    # Question 2
    community_mat_lastfm = Girvan_Newman(nodes_connectivity_list_lastfm)

    # Question 3
    visualise_dendogram(community_mat_lastfm)

    # Question 4
    graph_partition_louvain_lastfm = louvain_one_iter(
        nodes_connectivity_list_lastfm)
