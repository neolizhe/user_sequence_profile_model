# Construct coding tree
from typing import List, Annotated
from graphviz import Digraph
from random import sample
import numpy as np


class Node:
    def __init__(self, value: int = None, tag: str = None):
        self.value = value
        self.child_list = []  # 升序排序
        self.tag = tag
        self.trace_cnt = 1

    def get_child_by_index(self, id):
        return self.child_list[id]

    def check_child(self, child: None):
        if not self.child_list:
            return False
        for n in self.child_list:
            if child.value == n.value:
                return True
        return False

    def add_child(self, child):
        insert_index = -1
        child.tag = self.tag + '_' + str(child.value)
        if not self.child_list:
            self.child_list.append(child)
        else:
            for index in range(len(self.child_list)):
                if child.value < self.child_list[index].value:
                    self.child_list.insert(index, child)
                    insert_index = index
                    break
                elif child.value == self.child_list[index].value:
                    insert_index = index
                    break
            if self.child_list[-1].value < child.value:
                self.child_list.append(child)
        return insert_index

    def trim_child(self, ratio: float):
        trace_cnts = [x.trace_cnt for x in self.child_list]
        trimed_child = []
        if ratio < 1:
            point = np.percentile(trace_cnts, int(ratio * 100))
        else:
            trace_cnts = sorted(trace_cnts, reverse=True)
            if int(ratio) >= len(trace_cnts):
                point = 0
            else:
                point = trace_cnts[int(ratio)]
        trimed_child = [x for x in self.child_list if x.trace_cnt > point]
        self.child_list = trimed_child
        return


class TraceTree:
    def __init__(self, root_value: int):
        self.root = Node(root_value, tag="0")
        self.max_height = 0

    def mark_trace_start_pos_list(self, value: int):
        possible_pos_list = []
        single_pos_path = []
        head = self.root

        def _recrusive_mark_start(head, value, possible_pos_list, single_pos_path):
            if not head.child_list:
                # single_pos_path.clear()
                return
            for index in range(len(head.child_list)):
                single_pos_path.append(index)
                if value == head.child_list[index].value:
                    possible_pos_list.append(single_pos_path)
                _recrusive_mark_start(head.child_list[index], value, possible_pos_list, single_pos_path)
                single_pos_path.pop()
            return

        _recrusive_mark_start(head, value, possible_pos_list, single_pos_path)
        return possible_pos_list

    def generate_child_tree(self, head, trace):
        for t in trace:
            n = Node(t)
            insert_pos = head.add_child(n)
            head = head.child_list[insert_pos]
        return

    def get_leaf_node_by_path(self, path: List[int]):
        head = self.root
        for p in path:
            head = head.get_child_by_index(p)
        return head

    def check_trace_match_lens(self, head, trace):
        def _recrusive_match(head, trace_index):
            if trace_index >= len(trace):
                return 0

            match_lens = 0
            if head.value == trace[trace_index]:
                for child in head.child_list:
                    tmp_lens = _recrusive_match(child, trace_index + 1) + 1
                    match_lens = tmp_lens if tmp_lens > match_lens else match_lens

            return match_lens

        return _recrusive_match(head, 0)

    def add_trace(self, trace: List[int]):
        pos_list = self.mark_trace_start_pos_list(trace[0])
        if not pos_list:
            self.generate_child_tree(self.root, trace)
            return
        best_trace_index = -1
        max_lens = 0
        for path_index in range(len(pos_list)):
            head = self.get_leaf_node_by_path(pos_list[path_index])
            match_lens = self.check_trace_match_lens(head, trace)
            if match_lens > max_lens:
                best_trace_index = path_index
                max_lens = match_lens
        best_path = pos_list[best_trace_index]
        head = self.get_leaf_node_by_path(best_path)
        self.generate_child_tree(head, trace[max_lens:])
        return

    def generate_full_tree(self, df: pds.DataFrame):
        trace_list = df.id_seq.values
        for raw_trace in trace_list:
            trace = [int(x) for x in raw_trace.split("_")]
            self.add_trace(trace)
        return

    def update_trace_cnt(self):
        if not self.root.child_list:
            return

        def _recrusive_update_trace_cnt(head):
            if not head.child_list:
                return 1
            res_cnt = 0
            new_child_list = []
            for child in head.child_list:
                child_cnt = _recrusive_update_trace_cnt(child)
                child.trace_cnt = child_cnt
                new_child_list.append(child)
                res_cnt += child_cnt
            head.child_list = new_child_list
            return res_cnt

        root_total_traces = _recrusive_update_trace_cnt(self.root)
        self.root.trace_cnt = root_total_traces
        return

    def trim_tree(self, retain_ratio=0.2):
        # 保留每层样本数最多的top k default = 20%
        self.update_trace_cnt()

        def _recrusive_trim(head):
            if not head.child_list:
                return
            head.trim_child(retain_ratio)
            for child in head.child_list:
                _recrusive_trim(child)

        _recrusive_trim(self.root)
        return

    def display_tree(self, save_path=None, head=None):
        colors = ["pink", "red", "tomato", "orange", "yellow", "green", "skyblue", "purple"]
        plt = Digraph(comment="Tree", format="svg")
        plt.attr('graph', fixedsize='true', size='50,50')
        plt.attr('node', shape='circle')
        plt.attr(nodesep='0.02')

        # plt.attr(fontsize='1')
        def _print_node(node, color_level):
            color_index = color_level % len(colors)
            if len(node.child_list) > 1:
                # Set line color ordered by trace_cnt
                ilist = sorted(node.child_list, key=lambda x: x.trace_cnt, reverse=False)
                c_map = {ilist[index].tag: colors[index] for index in range(len(ilist))}
                for child in node.child_list:
                    # print(child.tag,child.value)
                    v = event_map_r[str(child.value)]
                    plt.node(child.tag, v,
                             # group=node.tag +'_' +str(color_level),
                             color=colors[color_index],
                             margin='0',
                             width=str(0.5 / color_level),
                             height=str(0.5 / color_level),
                             fontsize=str(20 / color_level))
                    if color_level > 0:
                        plt.edge(node.tag, child.tag,
                                 penwidth=str(child.trace_cnt / 20),
                                 arrowhead='none',
                                 color=c_map[child.tag])
                    _print_node(child, color_level + 1)
            return

        # print(self.root.tag,self.root.value)
        plt.node(head.tag, event_map_r[str(head.value)], color=colors[0])
        _print_node(head, 1)
        plt.view()
        if save_path:
            plt.render(save_path)
        return
