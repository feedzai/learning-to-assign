import itertools
from copy import deepcopy

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

def random_assignment(
        X, capacity, random_seed
):
    assignments = np.array(list(itertools.chain.from_iterable(
        [[expert_id] * expert_capacity for expert_id, expert_capacity in capacity.items()]
    )))
    np_rng = np.random.default_rng(seed=random_seed)
    np_rng.shuffle(assignments)

    assignments = pd.Series(assignments, index=X.index)

    return assignments

def optimal_batch_assignment(
        original_index, loss_df, index_col, assignment_col, cost_col, capacity,
):
    cost_matrix_df = loss_df.pivot(index=assignment_col, columns=index_col, values=cost_col)
    for d in capacity:
        if capacity[d] == 0:
            cost_matrix_df = cost_matrix_df.drop(index=d)

    cost_matrix = cost_matrix_df.values
    num_workers, num_tasks = cost_matrix.shape
    workers = list(cost_matrix_df.index)

    model = cp_model.CpModel()

    x = []
    for i in range(num_workers):
        t = []
        for j in range(num_tasks):
            t.append(model.NewBoolVar(f'x[{i},{j}]'))
        x.append(t)

    # capacity constraints
    for i in range(num_workers):
        model.Add(sum([x[i][j] for j in range(num_tasks)]) == capacity[workers[i]])

    # Each task is assigned to exactly one worker.
    for j in range(num_tasks):
        model.AddExactlyOne(x[i][j] for i in range(num_workers))

    objective_terms = []
    for i in range(num_workers):
        for j in range(num_tasks):
            objective_terms.append(cost_matrix[i, j] * x[i][j])
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True
    status = solver.Solve(model)

    if not status == cp_model.OPTIMAL and not status == cp_model.FEASIBLE:
        print('Solution not found!')
        return None

    assignments_list = list()
    for j in range(num_tasks):
        for i in range(num_workers):
            if solver.BooleanValue(x[i][j]):
                assignments_list.append(i)

    assignments_enc = pd.Series(assignments_list, index=original_index)
    assignments = assignments_enc.map(dict(zip(range(len(workers)), workers)))

    return assignments


def optimal_individual_assignment(original_index, loss_df, index_col, assignment_col, cost_col):
    # this method is much faster than using groupby() with idxmix()
    # index order recuperated afterwards
    assignments_df = (
        loss_df
        .sort_values(by=cost_col, ascending=True)
        .groupby(index_col)  # maintains sort order
        .head(1)
        .drop(columns=cost_col)
    )

    # recuperate original index order
    assignments = (
        pd.DataFrame(original_index, columns=[index_col])
        .merge(assignments_df, on=index_col)
        [assignment_col]
        .to_numpy()
    )

    return assignments

def case_by_case_assignment(
        original_index, loss_df, index_col, assignment_col, cost_col, capacity,
):
    c = deepcopy(capacity)
    assignments = list()
    cost_matrix = loss_df.pivot(index=assignment_col, columns=index_col, values=cost_col).T
    for ix, row in cost_matrix.iterrows():
        ascending_cost = row.sort_values(ascending=True)
        for expert in ascending_cost.index:
            if c[expert] > 0:
                assignments.append(expert)
                c[expert] -= 1
                break

    assignments = pd.Series(assignments, index=original_index)

    return assignments
