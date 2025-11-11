# Module Name: TSPSEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the constructive heuristic for Traveling Salseman Problem (TSP).
#              Given a set of locations,
#              the goal is to find optimal route to travel all locations and back to start point
#              while minimizing the total travel distance.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - problem_size: Number of customers to serve: int (default: 50).
#
# 
# References:
#   - Fei Liu, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang. 
#     "Algorithm Evolution using Large Language Model." arXiv preprint arXiv:2311.15249 (2023).
# 
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

import pickle
from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tsp_construct_set.get_instance import GetData
from llm4ad.task.optimization.tsp_construct_set.template import template_program, task_description

__all__ = ['TSPFMEvaluation']


class TSPSEvaluation(Evaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self,
                 timeout_seconds=30,
                 datasets=None,
                 return_list = False,
                 **kwargs):

        """
            Args:
                None
            Raises:
                AttributeError: If the data key does not exist.
                FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.return_list = return_list
        self.n_instance = 100000
        # getData = GetData(self.n_instance, problem_size)
        # self._datasets = getData.generate_instances()
        self._datasets = []
        for dataset in datasets:
            dataset = pickle.load(open(dataset, 'rb'))
            self._datasets.extend(dataset)
            print(f"load dataset from {dataset}")

        print(f"{len(self._datasets)} instances loaded")
        # dataset = pickle.load(open('./dataset_tsp_200_256.pkl', 'rb'))


    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def tour_cost(self, instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self, instance):
        instance = np.array(instance)
        n = len(instance)
        neighborhood_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            distances = np.linalg.norm(instance[i] - instance, axis=1)
            sorted_indices = np.argsort(distances)  # sort indices based on distances
            neighborhood_matrix[i] = sorted_indices

        return neighborhood_matrix

    def evaluate(self, eva: callable) -> float:


        dis = []
        n_ins = 0

        for instance, distance_matrix, baseline in self._datasets:

            problem_size = len(instance)

            # get neighborhood matrix
            neighbor_matrix = self.generate_neighborhood_matrix(instance)

            destination_node = 0

            current_node = 0

            route = np.zeros(problem_size)
            # print(">>> Step 0 : select node "+str(instance[0][0])+", "+str(instance[0][1]))
            for i in range(1, problem_size - 1):

                near_nodes = neighbor_matrix[current_node][1:]

                mask = ~np.isin(near_nodes, route[:i])

                unvisited_near_nodes = near_nodes[mask]

                next_node = eva(current_node, destination_node, unvisited_near_nodes, distance_matrix)

                if next_node in route:
                    # print("wrong algorithm select duplicate node, retrying ...")
                    return None

                current_node = next_node

                route[i] = current_node

            mask = ~np.isin(np.arange(problem_size), route[:problem_size - 1])

            last_node = np.arange(problem_size)[mask]

            current_node = last_node[0]

            route[problem_size - 1] = current_node

            LLM_dis = self.tour_cost(instance, route, problem_size)

            gap = (LLM_dis-baseline)/baseline

            dis.append(-gap)

            n_ins += 1
            if n_ins == self.n_instance:
                break
            # self.route_plot(instance,route,self.oracle[n_ins])

        #ave_dis = np.average(dis)
        if self.return_list:
            return dis
        else:
        # print("average dis: ",ave_dis)
            return np.average(dis)
        #return dis


if __name__ == '__main__':
    import sys

    print(sys.path)

    import numpy as np


    def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray,
                         distance_matrix: np.ndarray) -> int:
        """
        Design a novel algorithm to select the next node in each step.

        Args:
        current_node: ID of the current node.
        destination_node: ID of the destination node.
        unvisited_nodes: Array of IDs of unvisited nodes.
        distance_matrix: Distance matrix of nodes.

        Return:
        ID of the next node to visit.
        """
        current_distances = distance_matrix[current_node, unvisited_nodes]
        destination_distances = distance_matrix[unvisited_nodes, destination_node]
        average_unvisited_distance = np.mean(distance_matrix[unvisited_nodes][:, unvisited_nodes], axis=1)

        # Calculate scores based on the adjusted ratio
        scores = (current_distances / (1 + destination_distances)) + 0.5 * (1 / (average_unvisited_distance + 1))
        next_node_index = np.argmin(scores)  # Select node with minimum score

        return unvisited_nodes[next_node_index]


    tsp = TSPSEvaluation()
    dis = tsp.evaluate_program('_', select_next_node)
    print(dis)
