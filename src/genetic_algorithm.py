from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Callable, List, Optional, Sequence, Union

import pygad


Gene = Union[int, float]
FitnessFunc = Callable[..., float]


@dataclass
class GAResult:
    best_individual: List[Gene]
    best_fitness: float
    history: List[float]


class GeneticAlgorithm:
    def __init__(
        self,
        fitness_func: FitnessFunc,
        num_genes: int,
        population_size: int = 20,
        gene_space: Optional[Sequence[Gene]] = None,
        gene_type: Union[str, type] = int,
        gene_min: float = 0.0,
        gene_max: float = 1.0,
        num_parents_mating: Optional[int] = None,
        parent_selection_type: str = "tournament",
        crossover_type: str = "single_point",
        mutation_type: str = "random",
        mutation_percent_genes: float = 10.0,
        keep_elitism: int = 1,
        keep_parents: int = 0,
        random_seed: Optional[int] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        self.fitness_func = fitness_func
        self.num_genes = int(num_genes)
        self.population_size = int(population_size)
        self.gene_space = list(gene_space) if gene_space is not None else None
        self.gene_type = gene_type.__name__ if gene_type in (int, float) else gene_type
        self.gene_min = gene_min
        self.gene_max = gene_max
        self.num_parents_mating = num_parents_mating or max(2, self.population_size // 2)
        self.parent_selection_type = parent_selection_type
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        self.keep_elitism = keep_elitism
        self.keep_parents = keep_parents
        self.random_seed = random_seed
        self.maximize = maximize
        self.extra_kwargs = kwargs

    def _fitness_wrapper(self, ga_instance, solution, solution_idx):
        try:
            params = len(inspect.signature(self.fitness_func).parameters)
        except (TypeError, ValueError):
            params = 1

        if params <= 1:
            score = self.fitness_func(solution)
        elif params == 2:
            score = self.fitness_func(solution, solution_idx)
        else:
            score = self.fitness_func(ga_instance, solution, solution_idx)
        return float(score)

    def run(self, num_generations: int) -> GAResult:
        ga = pygad.GA(
            num_generations=int(num_generations),
            num_parents_mating=self.num_parents_mating,
            fitness_func=self._fitness_wrapper,
            sol_per_pop=self.population_size,
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            gene_type=self.gene_type,
            init_range_low=self.gene_min,
            init_range_high=self.gene_max,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            mutation_type=self.mutation_type,
            mutation_percent_genes=self.mutation_percent_genes,
            keep_elitism=self.keep_elitism,
            keep_parents=self.keep_parents,
            random_seed=self.random_seed,
            save_best_solutions=True,
            suppress_warnings=True,
            **self.extra_kwargs,
        )
        ga.run()
        best_solution, best_fitness, _ = ga.best_solution()
        history = [float(f) for f in getattr(ga, "best_solutions_fitness", [])]
        if not history:
            history = [float(best_fitness)]
        if not self.maximize:
            best_fitness = -best_fitness
            history = [-value for value in history]
        return GAResult(list(best_solution), float(best_fitness), history)


def optimize_cut_layer(
    fitness_func: FitnessFunc,
    valid_cut_layers: Sequence[int],
    population_size: int = 20,
    num_generations: int = 30,
    random_seed: Optional[int] = None,
    **kwargs,
) -> GAResult:
    ga = GeneticAlgorithm(
        fitness_func=fitness_func,
        num_genes=1,
        population_size=population_size,
        gene_space=list(valid_cut_layers),
        gene_type=int,
        random_seed=random_seed,
        **kwargs,
    )
    result = ga.run(num_generations=num_generations)
    result.best_individual = [int(result.best_individual[0])]
    return result


__all__ = ["GeneticAlgorithm", "GAResult", "optimize_cut_layer"]
