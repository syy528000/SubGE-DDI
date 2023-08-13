import torch
from subgraph_extraction.HK_random import HKrandom


def get_control_npathHK_all(n_path, nodes, edges):
        # pdb.set_trace()
        # get control path by maximum matching
        # nodes=[0,1,2,3,4,5]
        # edges=[(0,101,1),(1,102,2),(1,103,3),(2,104,4),(3,105,4),(2,106,5)]
        allpath=[]
        max_path_len = 0
        for i in range(n_path):
            hk = HKrandom(nodes, edges)
            path, max_len=hk.HK()
            allpath.extend(path)
            if max_len > max_path_len:
                max_path_len = max_len
            # pdb.set_trace()
        # allpath (source1,rel_1,target1)(target1,rel_2,target2)...
        # pdb.set_trace()
        alllen_path = 0
        newallpath = []
        for path in allpath:
            if path not in newallpath:
                newallpath.append(path)
                alllen_path = alllen_path + len(path)

        max_path_len=max_path_len+1
        all_node_list=[]
        all_rel_list=[]
        all_node_list_mask = []
        all_rel_list_mask = []
        edge_path=[]
        for path in newallpath:
            node_list = []
            rel_list = []
            node_list_mask = []
            rel_list_mask = []
            node_list.append(path[0][0])
            node_list_mask.append(1)
            for edge in path:
                # pdb.set_trace()
                edge_path.append([edge[2],edge[0]])  # adj is reverse with original edges
                node_list.append(edge[2])
                rel_list.append(edge[1])
                node_list_mask.append(1)
                rel_list_mask.append(1)
            # pdb.set_trace()
            for i_node in range(len(node_list),max_path_len):
                node_list.append(-1)
                rel_list.append(-1)
                node_list_mask.append(0)
                rel_list_mask.append(0)
            all_node_list.append(node_list)
            all_rel_list.append(rel_list)
            all_node_list_mask.append(node_list_mask)
            all_rel_list_mask.append(rel_list_mask)

        # pdb.set_trace()
        edge_path = torch.LongTensor(edge_path)
        all_node_list=torch.LongTensor(all_node_list)
        all_rel_list = torch.LongTensor(all_rel_list)
        all_node_list_mask = torch.LongTensor(all_node_list_mask)
        all_rel_list_mask = torch.LongTensor(all_rel_list_mask)
        return edge_path,all_node_list,all_node_list_mask,all_rel_list,all_rel_list_mask,max_path_len