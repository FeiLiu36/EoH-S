from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from ...base import *
import traceback

class Population:
    def __init__(self, pop_size, generation=0,top_k=5, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation
        self._clusters = []
        self._k = top_k

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def survival(self):
        pop = self._population + self._next_gen_pop
        pop = sorted(pop, key=lambda f: np.mean(f.score), reverse=True)
        self._population = pop[:self._pop_size]
        self._next_gen_pop = []
        self._generation += 1

    def update(self):

        self._lock.acquire()
        try:
            self.survival_set()
            #self._generation += 1
            #print(f'generation: {self._generation}')
        except Exception as e:
            print(e)
            # traceback.print_exc()
            return
        finally:
            self._lock.release()

    def survival_set(self):
        pop = self._population + self._next_gen_pop

        # Print initial population size
        print(f"Initial combined population size: {len(pop)}")

        pop = [
            f for f in pop
            if f.score is not None  # Exclude None
               and (
                       (isinstance(f.score, list) and not any(math.isinf(x) for x in f.score))  # List case: no infs
                       or (not isinstance(f.score, list) and not math.isinf(f.score))  # Single value case: not inf
               )
        ]

        # Print filtered population size
        print(f"Population after filtering None/inf scores: {len(pop)}")

        # Extract scores for each individual (assuming score is a list)
        score_lists = [ind.score for ind in pop]

        # Step 1: Sort by average score (descending) to get initial ordering
        avg_scores = [sum(scores) / len(scores) for scores in score_lists]
        sorted_indices = sorted(range(len(pop)), key=lambda i: -avg_scores[i])

        # Print top 5 average scores
        print("\nTop 5 individuals by average score:")
        for i in sorted_indices[:5]:
            print(f"Index {i}: Avg score {avg_scores[i]:.4f}, Scores: {pop[i].score}")

        survivors = []
        selected_indices = []

        if not sorted_indices:
            print("No valid individuals left after filtering!")
            self._population = []
            self._next_gen_pop = []
            self._generation += 1
            return

        # Step 2: Select first individual (best average)
        first_idx = sorted_indices[0]
        survivors.append(pop[first_idx])
        selected_indices.append(first_idx)
        print(f"\nSelected first survivor (best average): Index {first_idx}, Scores: {pop[first_idx].score}")

        # Step 3: Select remaining individuals one by one
        remaining_indices = [i for i in sorted_indices if i != first_idx]

        while len(survivors) < self._pop_size and remaining_indices:
            best_improvement = -float('inf')
            best_idx = None

            # Current combined scores (max of all selected individuals)
            current_combined = [max(survivor.score[i] for survivor in survivors)
                                for i in range(len(survivors[0].score))]

            print(f"\nCurrent combined scores: {current_combined}")
            print(f"Looking for next survivor among {len(remaining_indices)} candidates...")

            for candidate_idx in remaining_indices:
                candidate = pop[candidate_idx]

                # Calculate new combined scores if we add this candidate
                new_combined = [max(current_combined[i], candidate.score[i])
                                for i in range(len(current_combined))]

                # Calculate improvement (sum of improvements across all dimensions)
                improvement = sum(new_combined) - sum(current_combined)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_idx = candidate_idx

            if best_idx is not None:
                survivors.append(pop[best_idx])
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                print(
                    f"Selected survivor: Index {best_idx}, Improvement: {best_improvement:.4f}, Scores: {pop[best_idx].score}")
            else:
                # No improvement found, just add the next best average
                next_best = remaining_indices.pop(0)
                survivors.append(pop[next_best])
                print(
                    f"No improvement found, selecting next best average: Index {next_best}, Scores: {pop[next_best].score}")

        # Print final selected survivors
        print("\nFinal selected survivors:")
        for i, survivor in enumerate(survivors):
            print(f"Survivor {i + 1}: Scores: {survivor.score}, Avg: {sum(survivor.score) / len(survivor.score):.4f}")

        self._population = survivors[:self._pop_size]
        self._next_gen_pop = []
        self._generation += 1
        print(f"\nGeneration {self._generation} survival completed. Population size: {len(self._population)}")

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = float('-inf')
                # register to next_gen
            else:
                self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                self.survival()
        except Exception as e:
            return
        finally:
            self._lock.release()


    def register_function_set(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'

        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = float('-inf')
                # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            print(f"next pop size = {len(self._next_gen_pop)} / {self._pop_size}")
            if len(self._next_gen_pop) >= self._pop_size:
                self.survival_set()
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self) -> Function:

        # Filter out invalid functions (None scores or infinite scores)
        funcs = [
            f for f in self._population
            if f.score is not None  # Exclude None
               and (
                       (isinstance(f.score, list) and not any(math.isinf(x) for x in f.score))  # List case: no infs
                       or (not isinstance(f.score, list) and not math.isinf(f.score))  # Single value case: not inf
               )
        ]

        p = [1 / (r + len(funcs)) for r in range(len(funcs))]
        p = np.array(p)
        p = p / np.sum(p)

        return np.random.choice(funcs, p=p)


    def selection_from_k(self, n: int = 2) -> list[Function]:
        # Filter out invalid functions (None scores or infinite scores)
        funcs = [
            f for f in self._population
            if f.score is not None  # Exclude None
               and (
                       (isinstance(f.score, list) and not any(math.isinf(x) for x in f.score))  # List case: no infs
                       or (not isinstance(f.score, list) and not math.isinf(f.score))  # Single value case: not inf
               )
        ]

        # If there are fewer than k functions, use all available
        k = self._k

        k = min(k,len(funcs))

        # Get top k functions
        top_k = funcs[:k]

        # Calculate ranks (1-based) for probability distribution
        ranks = [k - i for i in range(k)]  # [k, k-1, ..., 1]
        p = np.array([1 / r for r in ranks])  # Inverse of rank
        p = p / np.sum(p)  # Normalize to probabilities

        # Select n functions from top k without replacement
        selected = np.random.choice(top_k, size=min(n, k), p=p, replace=False)

        return list(selected)


    def select_complementary_pair(self) -> Tuple[Function, Function]:
        """
        Selects a pair of functions from the top k based on their complementarity,
        measured by Manhattan distance between their score lists.

        Returns:
            A tuple of two functions (the selected complementary pair)

        Raises:
            ValueError: If fewer than 2 valid functions are available
        """
        # Filter out invalid functions (None scores or infinite scores)
        valid_funcs = [
            f for f in self._population
            if f.score is not None  # Exclude None
               and (
                       (isinstance(f.score, list) and not any(math.isinf(x) for x in f.score))  # List case: no infs
                       or (not isinstance(f.score, list) and not math.isinf(f.score))  # Single value case: not inf
               )
        ]

        # Check we have at least 2 functions
        if len(valid_funcs) < 2:
            raise ValueError("Need at least 2 valid functions to select a pair")

        # Get top k functions
        k = self._k
        k = min(k, len(valid_funcs))
        top_k = valid_funcs[:k]

        # Generate all possible unique pairs from top_k
        pairs = []
        for i in range(len(top_k)):
            for j in range(i + 1, len(top_k)):
                pairs.append((top_k[i], top_k[j]))

        # Calculate Manhattan distances for all pairs
        distances = []
        for f1, f2 in pairs:
            # Convert single scores to lists for uniform handling
            score1 = [f1.score] if not isinstance(f1.score, list) else f1.score
            score2 = [f2.score] if not isinstance(f2.score, list) else f2.score

            # Ensure scores have same length (pad with zeros if needed)
            max_len = max(len(score1), len(score2))
            score1 = score1 + [0] * (max_len - len(score1))
            score2 = score2 + [0] * (max_len - len(score2))

            # Calculate Manhattan distance
            distance = sum(abs(a - b) for a, b in zip(score1, score2))
            distances.append(distance)

        # Create a list of (distance, pair) tuples and sort by distance
        distance_pair_tuples = list(zip(distances, pairs))
        # Sort by the first element (distance) in each tuple
        sorted_distance_pairs = sorted(distance_pair_tuples, key=lambda x: x[0], reverse=True)
        # Extract just the pairs in sorted order
        ranked_pairs = [pair for _, pair in sorted_distance_pairs]

        # Create probability distribution based on ranks (higher rank = higher probability)
        ranks = [len(ranked_pairs) - i for i in range(len(ranked_pairs))]  # [k, k-1, ..., 1]
        p = np.array([1 / r for r in ranks])  # Inverse of rank
        p = p / np.sum(p)  # Normalize to probabilities

        # Select one pair based on the probability distribution
        selected_idx = np.random.choice(len(ranked_pairs), p=p)
        return ranked_pairs[selected_idx]
