import heapq
import math
import logging
import random
import numpy as np


from src.datas.make_data.trans_torch import StackedTorchDataPackage, StackedDataSet
import matplotlib.pyplot as plt

from src.library.cache_io import get_root_path
from src.library.custom_exceptions import DataPartitionError
from src.datas.partition.partition_unit import Partition, equal_division
from src.library.logger import create_logger


logger = create_logger()


class HorizontalPartition(Partition):
    def __init__(self, name, partition):
        """
        Parent class according to sample data.
        """
        self.partition = partition
        self.data_distribution = self.print_data_distribution()
        super(HorizontalPartition, self).__init__(name, partition)

    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset[p][0], targets=dataset[p][1])
            for i, p in enumerate(self.partition)
        ]

    def print_data_distribution(self):
        """
        Return the distribution of data categories for each node through cumulative bar charts.
        """
        data_distribution = {}
        data_y = np.array([label.item() for _, (_, label) in enumerate(self.dataset)])
        for index, data_idx in enumerate(self.partition):
            unique_values, counts = np.unique(data_y[data_idx], return_counts=True)
            distribution = {unique_values[i]: counts[i] for i in range(len(unique_values))}
            data_distribution[index] = distribution
        logger.debug("The categorical dataset is divided into distributions at each node as follows: "
                     + str(data_distribution))
        return data_distribution

    def draw_data_distribution(self, new_labels=None):
        """
        Draw data distributions for all nodes,
        showing the distribution of data categories for each node through cumulative bar charts.
        """
        labels = [i for i in range(1, len(self.partition) + 1)]
        class_list = sorted(list(set([label.item() for _, label in self.dataset])))
        class_cnt = len(class_list)
        data = [[] for _ in range(class_cnt)]

        for j in class_list:
            data[j] = np.array([self.data_distribution[x][j] if self.data_distribution[x].get(j) else 0
                                for x in self.data_distribution])
        sum_data = sum(data)
        y_max = max(sum_data)
        x = range(len(labels))
        width = 0.35

        # Initialize bottom_y element 0
        bottom_y = np.array([0] * len(labels))

        fig, ax = plt.subplots()
        for i, y in enumerate(data):
            ax.bar(x, y, width, bottom=bottom_y, label=class_list[i])
            bottom_y = bottom_y + y

        # Add Legend
        plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

        # for a, b, i in zip(x, sum_data, range(len(x))):
        #     plt.text(a, int(b * 1.03), "%d" % sum_data[i], ha='center')

        # Setting the title and axis labels
        if new_labels is None:
            ax.set_title(self.name + ' data distribution')
            ax.set_xlabel('Nodes')
            ax.set_ylabel('Sample number')
            plt.xticks(x)
            name = ""
        else:
            # ax.set_title(self.name + ' data distribution')
            from matplotlib import font_manager
            font = font_manager.FontProperties(fname="/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf")
            ax.set_xlabel(new_labels["xlabel"], fontproperties=font, fontsize=13)
            ax.set_ylabel(new_labels["ylabel"], fontproperties=font, fontsize=13)
            plt.xticks(x, fontproperties=font)
            name = new_labels["name"]

        # Adjust chart layout to prevent annotations from obscuring chart content
        plt.tight_layout()

        plt.grid(True, linestyle=':', alpha=0.6)
        # Adjust the length of the vertical coordinate
        plt.ylim(0, int(y_max * 1.1))

        # show picture
        plt.show()
        picture_name = "{}_{}_{}.png".format(self.name, self.node_cnt, name)
        path_list = ["record", "report", "picture", "partition"]
        data_root = ""#"../../"
        save_path = get_root_path(picture_name, path_list, data_root, create_if_not_exist=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()


class EmptyPartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw):
        self.dataset = dataset
        self.node_cnt = node_cnt
        partition = [[] for _ in range(node_cnt)]
        super(EmptyPartition, self).__init__('EmptyPartition', partition)


class SuccessivePartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        """
        Successive segmentation divides the dataset to individual nodes.

        This works for datasets with continuous labels as well.

        data separation, with the form of [d(0), d(1), d(2), ..., d(node_cnt)]
        Node i have the dataset indexed by [d(i), d(i+1))

        data partition, with the form of
        [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        Node i have the dataset indexed by [l(n), r(n))

        """
        self.dataset = dataset
        self.node_cnt = node_cnt
        separation = [(i * len(self.dataset)) // node_cnt for i in range(node_cnt + 1)]

        partition = [list(range(separation[i], separation[i + 1]))
                     for i in range(node_cnt)]
        super(SuccessivePartition, self).__init__('SuccessivePartition', partition)


class IIDPartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        """
        Successive segmentation divides the shuffle dataset to individual nodes

        data separation, with the form of [d(0), d(1), d(2), ..., d(node_cnt)]
        Node i have the dataset indexed by [d(i), d(i+1))

        data partition, with the form of
        [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        Node i have the dataset indexed by [l(n), r(n))
        """
        self.dataset = dataset
        self.node_cnt = node_cnt
        indexes = list(range(len(dataset)))
        random.shuffle(indexes)
        sep = [(i * len(dataset)) // node_cnt for i in range(node_cnt + 1)]

        partition = [[indexes[i] for i in range(sep[node], sep[node + 1])]
                     for node in range(node_cnt)]
        super(IIDPartition, self).__init__('IIDPartition', partition)


#每个人固定样本数的IID分类方式
class FIXIIDPartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        """
        Successive segmentation divides the shuffle dataset to individual nodes

        data separation, with the form of [d(0), d(1), d(2), ..., d(node_cnt)]
        Node i have the dataset indexed by [d(i), d(i+1))

        data partition, with the form of
        [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        Node i have the dataset indexed by [l(n), r(n))
        """
        self.dataset = dataset
        self.node_cnt = node_cnt
        # 固定每个节点分配的样本数量
        samples_per_node = 6000
        indexes = list(range(len(dataset)))
        random.shuffle(indexes)

        # 计算能分配的最大节点数
        max_node_cnt = min(node_cnt, len(dataset) // samples_per_node)
        # partition = [
        #     indexes[i * samples_per_node : (i + 1) * samples_per_node]
        #     for i in range(max_node_cnt)
        # ]
        sep = [i * samples_per_node for i in range(node_cnt + 1)]

        partition = [[indexes[i] for i in range(sep[node], sep[node + 1])]
                     for node in range(max_node_cnt)]
        super(FIXIIDPartition, self).__init__('FIXIIDPartition', partition)


class SharedData(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        self.dataset = dataset
        self.node_cnt = node_cnt
        partition = [list(range(len(dataset)))] * node_cnt
        super(SharedData, self).__init__('SharedData', partition)


class LabelSeparation(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw):
        """
        Divide the dataset into non-independent identical distributions, this method is from Zhao Xian Wu.
        This method has few adjustable parameters.

        Args:
            dataset: StackedDataSet, train dataset or test dataset
            node_cnt: number of label class
        """
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)
        self.node_cnt = node_cnt
        self.dataset = dataset
        # deal with the situation that class idx don't
        # locate in consecutive integers starting from zeros
        self.class_idx_dict = {
            label: idx for idx, label in enumerate(self.class_set)}

        if self.class_cnt < node_cnt:
            partition = self.partition_with_adequate_nodes()
        else:
            partition = self.partition_with_adequate_classes()
        super(LabelSeparation, self).__init__('LabelSeparation', partition)

    def partition_with_adequate_classes(self):
        """
        class_cnt >= node_cnt
        some nodes possess several classes

        [e.g]
        class_cnt = 5
        node_cnt = 4

        class 0,4 on node 0
        class 1 on node 1
        class 2 on node 2
        class 3 on node 3
        """
        partition = [[] for _ in range(self.node_cnt)]
        for data_idx, (_, label) in enumerate(self.dataset):
            node_idx = self.class_idx_dict[label.item()] % self.node_cnt
            partition[node_idx].append(data_idx)
        return partition

    def partition_with_adequate_nodes(self):
        """
        class_cnt < node_cnt
        some classes are allocated on different nodes

        [e.g]
        class_cnt = 5
        node_cnt = 8
        group_boundary = [0, 1, 3, 4, 6, 8]
        divide 8 nodes into 5 groups by
        0 | 1 | 2 3 | 4 5 | 6 7 |
        where the vertical line represent the corresponding `group_boundary`
        this means
        class 0 on node 0
        class 1 on node 1
        class 2 on node 2, 3
        class 3 on node 4, 5
        class 4 on node 6, 7
        """

        class_cnt = self.class_cnt
        node_cnt = self.node_cnt
        dataset = self.dataset

        # divide the nodes into `class_cnt` groups
        group_boundary = [(group_idx * node_cnt) // class_cnt
                          for group_idx in range(class_cnt)]
        # when a data is going to be allocated to `group_idx`-th groups,
        # it will be allocated to `insert_node_ptrs[group_idx]`-th node
        # then `insert_node_ptrs[group_idx]` increases by 1
        insert_node_ptrs = group_boundary.copy()
        group_boundary.append(node_cnt)

        partition = [[] for _ in range(node_cnt)]
        for data_idx, (_, label) in enumerate(dataset):
            # determine which group the data belongs to
            group_idx = self.class_idx_dict[label.item()]
            node_idx = insert_node_ptrs[group_idx]
            partition[node_idx].append(data_idx)
            # `insert_node_ptrs[group_idx]` increases by 1
            if insert_node_ptrs[group_idx] + 1 < group_boundary[group_idx + 1]:
                insert_node_ptrs[group_idx] += 1
            else:
                insert_node_ptrs[group_idx] = group_boundary[group_idx]
        return partition


class NonIIDSeparation(HorizontalPartition):
    def __init__(self, dataset, node_cnt, class_per_node=1, data_balance=True, *args, **kw):
        """
        Divide the dataset into non-independent identical distributions.

        Args:
            dataset: StackedDataSet, train dataset or test dataset
            node_cnt: number of label class
            class_per_node: class number of per node
            data_balance: Unbalance mean each node may have the difference size dataset,
                            balance means each node has the approximately same size dataset.
        """
        self.dataset = dataset
        self.node_cnt = node_cnt
        self.class_per_node = class_per_node

        # Get the label set and label number
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)

        # deal with the situation that class idx don't
        # locate in consecutive integers starting from zeros
        self.class_idx_dict = {
            label: idx for idx, label in enumerate(self.class_set)}

        # The number of label classes each node can have cannot be greater than the total number of classes
        if self.class_per_node > self.class_cnt:
            self.class_per_node = self.class_cnt

        # total no. of parts
        self.total_amount = self.class_per_node * self.node_cnt
        # no. of parts each class should be divided into
        self.parts_per_class = math.ceil(self.total_amount / self.class_cnt)

        if data_balance is False:
            partition = self.non_iid_unbalance()
        else:
            partition = self.non_iid_balance()

        super(NonIIDSeparation, self).__init__('NonIIDSeparation', partition)

    def non_iid_unbalance(self):
        """
        Unbalance mean each node may have the difference size dataset.

        Note:
            When class_cnt < node_cnt, this method may make data unavailable to some nodes and is not recommended.

        We use a licensing strategy.
        Each class is divided into a number of copies and then issued to the corresponding nodes in order of node order.

        [e.g]
        class_cnt = 5
        node_cnt = 5
        class_per_node = 2

        node 0 has class 0, 1
        node 1 has class 1, 2
        node 2 has class 2, 3
        node 3 has class 3, 4
        node 4 has class 4, 0

        [e.g] unbalance dataset example
        class_cnt = 6
        node_cnt = 5
        class_per_node = 2

        node 0 has class 0, 1, 5
        node 1 has class 1, 2, 5
        node 2 has class 2, 3
        node 3 has class 3, 4
        node 4 has class 4, 0

        """
        partition = [[] for _ in range(self.node_cnt)]
        flags = [0] * self.class_cnt

        if self.class_per_node*self.class_cnt < self.node_cnt:
            self.parts_per_class = int(self.node_cnt / (self.class_per_node*self.class_cnt))
            for i, (_, label) in enumerate(self.dataset):
                label = self.class_idx_dict[label.item()]
                if flags[label] != (self.parts_per_class - 1):
                    partition[(label*self.parts_per_class + flags[label]) % self.node_cnt].append(i)
                    flags[label] += 1
                else:
                    partition[(label*self.parts_per_class + self.parts_per_class - 1) % self.node_cnt].append(i)
                    flags[label] = 0
        else:
            for i, (_, label) in enumerate(self.dataset):
                label = self.class_idx_dict[label.item()]
                if flags[label] != (self.parts_per_class - 1):
                    partition[(label + flags[label]) % self.node_cnt].append(i)
                    flags[label] += 1
                else:
                    partition[(label + self.parts_per_class - 1) % self.node_cnt].append(i)
                    flags[label] = 0

        for tem in partition:
            if not tem:
                info = 'Some nodes has no data, please decrease the node_cnt or increase class_per_node.'
                logger.error(info)
                raise DataPartitionError(info)
        return partition

    def non_iid_balance(self):
        """
        Balance means each node has the approximately same size dataset.

        Partition dataset into multiple nodes based on label classes.
        Each node contains class_per_node classes, where class_per_node is the number of classes of a node.

        Note: Each class is divided into `ceil(class_per_node * self.node_cnt / class_cnt)` parts
            and each node chooses `class_per_node` parts from each class to construct its dataset.

        Return:
            list[list[]]: The partitioned data.

        [e.g]
        class_cnt = 10
        node_cnt = 5
        class_per_node = 2

        node 0 has class 0, 1
        node 1 has class 2, 3
        node 2 has class 4, 5
        node 3 has class 6, 7
        node 4 has class 8, 9
        """

        all_index = [[] for _ in range(self.class_cnt)]

        for i, (_, label) in enumerate(self.dataset):
            # get indexes for all data with current label i at index i in all_index
            label = self.class_idx_dict[label.item()]
            all_index[label].append(i)

        partition = [[] for _ in range(self.node_cnt)]

        class_map = {}
        parts_consumed = []
        for i in range(self.class_cnt):
            class_map[i], _ = equal_division(self.parts_per_class, all_index[i])
            heapq.heappush(parts_consumed, (0, i))
        for i in range(self.node_cnt):
            for j in range(self.class_per_node):
                class_chosen = heapq.heappop(parts_consumed)
                part_indexes = class_map[class_chosen[1]].pop(0)

                heapq.heappush(parts_consumed, (class_chosen[0] + 1, class_chosen[1]))
                partition[i].extend(part_indexes)
        return partition



# class NonIIDFIXSeparation(HorizontalPartition):
#     def __init__(self, dataset, node_cnt, class_per_node=2, data_balance=True, samples_per_node=3000, *args, **kw):
#         """
#         将数据集划分为非独立同分布的数据，同时可以控制每个节点的样本数量。

#         参数:
#             dataset: StackedDataSet, 训练数据集或测试数据集
#             node_cnt: 节点数量
#             class_per_node: 每个节点包含的类别数量
#             data_balance: True表示数据平衡，False表示数据不平衡
#             samples_per_node: 每个节点的样本数量，若为None则不限制样本数量
#         """
#         self.dataset = dataset
#         self.node_cnt = node_cnt
#         self.class_per_node = class_per_node
#         self.samples_per_node = samples_per_node

#         # 获取标签集合和标签数量
#         self.class_set = set([label.item() for _, label in dataset])
#         self.class_cnt = len(self.class_set)

#         # 处理标签索引不连续的情况
#         self.class_idx_dict = {
#             label: idx for idx, label in enumerate(self.class_set)}

#         # 每个节点能够包含的标签类别数量不能超过总的类别数
#         if self.class_per_node > self.class_cnt:
#             self.class_per_node = self.class_cnt

#         # 总的划分部分数
#         self.total_amount = self.class_per_node * self.node_cnt
#         # 每个类别需要划分的部分数
#         self.parts_per_class = math.ceil(self.total_amount / self.class_cnt)

#         if data_balance is False:
#             partition = self.non_iid_unbalance()
#         else:
#             partition = self.non_iid_balance()

#         super(NonIIDFIXSeparation, self).__init__('NonIIDFIXSeparation', partition)

#     def non_iid_unbalance(self):
#         """
#         非平衡数据划分，意味着每个节点的数据集大小可能不同。

#         返回:
#             list[list[]]: 划分后的数据。
#         """
#         partition = [[] for _ in range(self.node_cnt)]
#         flags = [0] * self.class_cnt

#         if self.class_per_node * self.class_cnt < self.node_cnt:
#             self.parts_per_class = int(self.node_cnt / (self.class_per_node * self.class_cnt))
#             for i, (_, label) in enumerate(self.dataset):
#                 if self.samples_per_node is not None and len(partition[i % self.node_cnt]) >= self.samples_per_node:
#                     continue
                
#                 label = self.class_idx_dict[label.item()]
#                 if flags[label] != (self.parts_per_class - 1):
#                     partition[(label * self.parts_per_class + flags[label]) % self.node_cnt].append(i)
#                     flags[label] += 1
#                 else:
#                     partition[(label * self.parts_per_class + self.parts_per_class - 1) % self.node_cnt].append(i)
#                     flags[label] = 0
#         else:
#             for i, (_, label) in enumerate(self.dataset):
#                 if self.samples_per_node is not None and len(partition[i % self.node_cnt]) >= self.samples_per_node:
#                     continue
                
#                 label = self.class_idx_dict[label.item()]
#                 if flags[label] != (self.parts_per_class - 1):
#                     partition[(label + flags[label]) % self.node_cnt].append(i)
#                     flags[label] += 1
#                 else:
#                     partition[(label + self.parts_per_class - 1) % self.node_cnt].append(i)
#                     flags[label] = 0

#         for tem in partition:
#             if not tem:
#                 info = '某些节点没有数据，请减少节点数量或增加每个节点的类别数量。'
#                 logger.error(info)
#                 raise DataPartitionError(info)
#         return partition

#     def non_iid_balance(self):
#         """
#         平衡数据划分，意味着每个节点的数据集大小大致相同。

#         返回:
#             list[list[]]: 划分后的数据。
#         """
#         all_index = [[] for _ in range(self.class_cnt)]

#         for i, (_, label) in enumerate(self.dataset):
#             label = self.class_idx_dict[label.item()]
#             all_index[label].append(i)

#         partition = [[] for _ in range(self.node_cnt)]

#         class_map = {}
#         parts_consumed = []
#         for i in range(self.class_cnt):
#             class_map[i], _ = equal_division(self.parts_per_class, all_index[i])
#             heapq.heappush(parts_consumed, (0, i))
        
#         for i in range(self.node_cnt):
#             current_sample_count = 0
#             while current_sample_count < self.samples_per_node if self.samples_per_node else True:
#                 for j in range(self.class_per_node):
#                     class_chosen = heapq.heappop(parts_consumed)
#                     part_indexes = class_map[class_chosen[1]].pop(0)
#                     current_sample_count += len(part_indexes)
                    
#                     if self.samples_per_node and current_sample_count > self.samples_per_node:
#                         part_indexes = part_indexes[:self.samples_per_node - (current_sample_count - len(part_indexes))]
                    
#                     heapq.heappush(parts_consumed, (class_chosen[0] + 1, class_chosen[1]))
#                     partition[i].extend(part_indexes)
                    
#                     if self.samples_per_node and current_sample_count >= self.samples_per_node:
#                         break
#         return partition



class NonIIDFIXSeparation(HorizontalPartition):
    def __init__(self, dataset, node_cnt, class_per_node=6, data_balance=True, samples_per_node=3000, *args, **kw):
        """
        将数据集划分为非独立同分布的数据，同时可以控制每个节点的样本数量。

        参数:
            dataset: StackedDataSet, 训练数据集或测试数据集
            node_cnt: 节点数量
            class_per_node: 每个节点包含的类别数量
            data_balance: True表示数据平衡，False表示数据不平衡
            samples_per_node: 每个节点的样本数量，若为None则不限制样本数量
        """
        self.dataset = dataset
        self.node_cnt = node_cnt
        self.class_per_node = class_per_node
        self.samples_per_node = samples_per_node

        # 获取标签集合和标签数量
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)

        # 处理标签索引不连续的情况
        self.class_idx_dict = {label: idx for idx, label in enumerate(self.class_set)}

        # 每个节点能够包含的标签类别数量不能超过总的类别数
        if self.class_per_node > self.class_cnt:
            self.class_per_node = self.class_cnt

        # 总的划分部分数
        self.total_amount = self.class_per_node * self.node_cnt
        # 每个类别需要划分的部分数
        self.parts_per_class = math.ceil(self.total_amount / self.class_cnt)

        if data_balance is False:
            partition = self.non_iid_unbalance()
        else:
            partition = self.non_iid_balance()

        super(NonIIDFIXSeparation, self).__init__('NonIIDFIXSeparation', partition)

    def non_iid_unbalance(self):
        """
        非平衡数据划分，意味着每个节点的数据集大小可能不同。

        返回:
            list[list[]]: 划分后的数据。
        """
        partition = [[] for _ in range(self.node_cnt)]
        flags = [0] * self.class_cnt

        if self.class_per_node * self.class_cnt < self.node_cnt:
            self.parts_per_class = int(self.node_cnt / (self.class_per_node * self.class_cnt))
            for i, (_, label) in enumerate(self.dataset):
                if self.samples_per_node is not None and len(partition[i % self.node_cnt]) >= self.samples_per_node:
                    continue

                label = self.class_idx_dict[label.item()]
                if flags[label] != (self.parts_per_class - 1):
                    partition[(label * self.parts_per_class + flags[label]) % self.node_cnt].append(i)
                    flags[label] += 1
                else:
                    partition[(label * self.parts_per_class + self.parts_per_class - 1) % self.node_cnt].append(i)
                    flags[label] = 0
        else:
            for i, (_, label) in enumerate(self.dataset):
                if self.samples_per_node is not None and len(partition[i % self.node_cnt]) >= self.samples_per_node:
                    continue

                label = self.class_idx_dict[label.item()]
                if flags[label] != (self.parts_per_class - 1):
                    partition[(label + flags[label]) % self.node_cnt].append(i)
                    flags[label] += 1
                else:
                    partition[(label + self.parts_per_class - 1) % self.node_cnt].append(i)
                    flags[label] = 0

        for tem in partition:
            if not tem:
                info = '某些节点没有数据，请减少节点数量或增加每个节点的类别数量。'
                logger.error(info)
                raise DataPartitionError(info)
        return partition

    def non_iid_balance(self):
        """
        平衡数据划分，意味着每个节点的数据集大小大致相同，每个类的样本量在节点中也尽量均匀。

        返回:
            list[list[]]: 划分后的数据。
        """
        all_index = [[] for _ in range(self.class_cnt)]

        for i, (_, label) in enumerate(self.dataset):
            label = self.class_idx_dict[label.item()]
            all_index[label].append(i)

        partition = [[] for _ in range(self.node_cnt)]
        class_map = {}
        parts_consumed = []

        # 计算每个类别需要划分的样本数目
        samples_per_class = self.samples_per_node // self.class_per_node if self.samples_per_node else None

        for i in range(self.class_cnt):
            class_map[i], _ = equal_division(self.parts_per_class, all_index[i])
            heapq.heappush(parts_consumed, (0, i))
        
        for i in range(self.node_cnt):
            current_sample_count = 0
            class_counts = {j: 0 for j in range(self.class_cnt)}
            
            while current_sample_count < self.samples_per_node if self.samples_per_node else True:
                for j in range(self.class_per_node):
                    class_chosen = heapq.heappop(parts_consumed)
                    part_indexes = class_map[class_chosen[1]].pop(0)
                    
                    if samples_per_class:
                        # 确保每个类别的样本数量均衡
                        available_samples = samples_per_class - class_counts[class_chosen[1]]
                        part_indexes = part_indexes[:available_samples]
                    
                    current_sample_count += len(part_indexes)
                    class_counts[class_chosen[1]] += len(part_indexes)

                    heapq.heappush(parts_consumed, (class_chosen[0] + 1, class_chosen[1]))
                    partition[i].extend(part_indexes)

                    if self.samples_per_node and current_sample_count >= self.samples_per_node:
                        break

        return partition




class DirichletNonIID(HorizontalPartition):
    def __init__(self, dataset, node_cnt, alpha=0.001, min_size=10, *args, **kw):
        self.dataset = dataset
        self.node_cnt = node_cnt
        self.alpha = alpha
        self.min_size = min_size

        # Get the label set and label number
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)

        # deal with the situation that class idx don't
        # locate in consecutive integers starting from zeros
        self.class_idx_dict = {
            label: idx for idx, label in enumerate(self.class_set)}

        partition = self.non_iid_dirichlet()
        super(DirichletNonIID, self).__init__('DirichletNonIID', partition)

    def non_iid_dirichlet(self):
        """Partition dataset into multiple clients following the Dirichlet process.

        Key parameters:
            alpha (float): The parameter for Dirichlet process simulation.
            min_size (int): The minimum number of data size of a client.

        Return:
            list[list[]]: The partitioned data.
        """

        current_min_size = 0
        data_size = len(self.dataset)

        all_index = [[] for _ in range(self.class_cnt)]
        for i, (_, label) in enumerate(self.dataset):
            # get indexes for all data with current label i at index i in all_index
            label = self.class_idx_dict[label.item()]
            all_index[label].append(i)

        partition = [[] for _ in range(self.node_cnt)]
        while current_min_size < self.min_size:
            partition = [[] for _ in range(self.node_cnt)]
            for k in range(self.class_cnt):
                idx_k = all_index[k]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.node_cnt))
                # using the proportions from dirichlet, only select those nodes having data amount less than average
                proportions = np.array(
                    [p * (len(idx_j) < data_size / self.node_cnt) for p, idx_j in zip(proportions, partition)])
                # scale proportions
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                partition = [idx_j + idx.tolist() for idx_j, idx in zip(partition, np.split(idx_k, proportions))]
                current_min_size = min([len(idx_j) for idx_j in partition])
        return partition


class DirichletFixNonIID(HorizontalPartition):
    def __init__(self, dataset, node_cnt, sample_per_node=2000, alpha=1, min_size=10, *args, **kw):
        self.dataset = dataset
        self.node_cnt = node_cnt
        self.sample_per_node = sample_per_node
        self.alpha = alpha
        self.min_size = min_size

        # Get the label set and label number
        self.class_set = set([label.item() for _, label in dataset])
        self.class_cnt = len(self.class_set)
        
        # Handle class index mapping if not consecutive
        self.class_idx_dict = {label: idx for idx, label in enumerate(self.class_set)}

        partition = self.non_iid_dirichlet()
        super(DirichletFixNonIID, self).__init__('DirichletFixNonIID', partition)

    def non_iid_dirichlet(self):
        """Partition dataset into multiple clients with a fixed sample size per node."""
        
        all_index = [[] for _ in range(self.class_cnt)]
        for i, (_, label) in enumerate(self.dataset):
            label = self.class_idx_dict[label.item()]
            all_index[label].append(i)
        
        partition = [[] for _ in range(self.node_cnt)]
        
        while True:
            partition = [[] for _ in range(self.node_cnt)]
            for k in range(self.class_cnt):
                idx_k = all_index[k]
                np.random.shuffle(idx_k)
                
                # Sample proportions from Dirichlet distribution
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.node_cnt))
                
                # Scale proportions based on sample size per node
                proportions = (proportions / proportions.sum()) * len(idx_k)
                proportions = np.cumsum(proportions).astype(int)[:-1]
                
                # Assign samples to nodes
                partition = [idx_j + idx.tolist() for idx_j, idx in zip(partition, np.split(idx_k, proportions))]
            
            # Check if all nodes meet the sample size requirement
            if all(len(idx_j) >= self.sample_per_node for idx_j in partition):
                break
        
        # Trim excess samples to ensure exact sample_per_node per client
        for i in range(self.node_cnt):
            np.random.shuffle(partition[i])
            partition[i] = partition[i][:self.sample_per_node]
        
        return partition


class VerticalPartition(Partition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        """
        data separation, with the form of [d(0), d(1), d(2), ..., d(n)]
        Node i have the dataset indexed by [d(i), d(i+1))

        data partition, with the form of
        [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        Node i have the dataset indexed by [l(n), r(n))
        """
        feature_dimension = dataset.features[0].nelement()
        separation = [(i * feature_dimension) // node_cnt
                      for i in range(node_cnt + 1)]

        partition = [list(range(separation[i], separation[i + 1]))
                     for i in range(node_cnt)]
        self.partition = partition
        super(VerticalPartition, self).__init__('VerticalPartition', partition)

    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset.features[:, p],
                           targets=dataset.targets)
            for i, p in enumerate(self.partition)
        ]
