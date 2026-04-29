#!/usr/bin/env python3
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_DIR / "code"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

if __name__ == "__main__":
    import argparse
    from mealpy.evolutionary_based.SHADE import L_SHADE
    import pandas as pd
    import random
    import torch
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    from data_proc.mol_preproc import TU_MUTAG_CONFIG, create_processor_list
    from gnn_.datasets_ import ProcessedInMemoryDataset
    from hpo.gen_model import (
        DEFOG_BOUNDS,
        DeFoGMolSearchProblem
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-o", "--output-file", type=str,
        help="file to store the final neural architecture configuration in"
    )
    common_parser.add_argument(
        "-g", "--generations", type=int,
        help="number of generations (or HPO/NAS epochs) to optimize the population for"
    )
    common_parser.add_argument(
        "-p", "--population-size", type=int,
        help="number of population members that participate in the optimiation"
    )
    common_parser.add_argument(
        "-e", "--epochs", type=int, default=100,
        help="number of (deep learning) epochs to train each model for (default: 100)"
    )
    common_parser.add_argument(
        "-b", "--batch-size", type=int, default=256,
        help="batch size for dataloader (default: 256)"
    )
    common_parser.add_argument(
        "-sv", "--no-of-solver-steps", type=int, default=100,
        help="number of discrete solver steps (default: 100)"
    )
    common_parser.add_argument(
        "-s", "--random-seed", type=int, default=1,
        help="random seed used for Python, NumPy and PyTorch (default: 1)"
    )
    parser = argparse.ArgumentParser(
        prog="mol_gen_nas.py",
        description="perform a naive neural architecture search of a molecule generator using metaheuristic optimization",
        parents=[common_parser]
    )

    subparsers = parser.add_subparsers(dest="model_type", required=True)

    defog_parent = argparse.ArgumentParser(add_help=False)
    defog_parser = subparsers.add_parser(
        name="defog",
        prog="mol_gen_nas.py defog",
        description="Use the DeFoG architecture (discrete flow matching over ndde/edge tokens)",
        parents=[common_parser, defog_parent]
    )

    cli_args = parser.parse_args()

    OUTPUT_FILE = Path(cli_args.output_file)
    GENERATIONS = cli_args.generations
    POP_SIZE = cli_args.population_size
    EPOCHS = cli_args.epochs
    BATCH_SIZE = cli_args.batch_size
    SOLV_STEPS = cli_args.no_of_solver_steps
    SEED = cli_args.random_seed

    random.seed(SEED)
    torch.manual_seed(SEED)

    EDGE_LIST_STR = list(TU_MUTAG_CONFIG["edge_dict"].keys())
    EDGE_LIST_RDK = list(TU_MUTAG_CONFIG["edge_dict"].values())
    KEK_EDGE_DICT = {
        n: b
        for n, b in TU_MUTAG_CONFIG["edge_dict"].items()
        if n != "aromatic"
    }
    KEK_EDGE_LIST_STR = list(KEK_EDGE_DICT.keys())
    KEK_EDGE_LIST_RDK = list(KEK_EDGE_DICT.values())

    trafos = create_processor_list(
        TU_MUTAG_CONFIG["node_list"], EDGE_LIST_STR, EDGE_LIST_RDK,
        TU_MUTAG_CONFIG["aromatic_idx"], TU_MUTAG_CONFIG["max_n_atoms"]
    )

    dataset = TUDataset(root=PROJECT_DIR / "data", name='MUTAG').to(DEVICE)
    proc_dataset = ProcessedInMemoryDataset(dataset, trafos).to(DEVICE)

    data_loader = DataLoader(proc_dataset, batch_size=BATCH_SIZE)

    if cli_args.model_type == "defog":
        problem = DeFoGMolSearchProblem(
        DEFOG_BOUNDS, data_set=proc_dataset, data_loader=data_loader,
        n_epochs=EPOCHS, n_solv_steps=SOLV_STEPS, n_samples=100, device=DEVICE
    )
    
    optimizer_hpo = L_SHADE(epoch=GENERATIONS, pop_size=POP_SIZE)
    optimizer_hpo.solve(problem, seed=SEED)

    best_hyperpars = optimizer_hpo.problem.decode_solution(
        optimizer_hpo.g_best.solution)
    print(best_hyperpars)
    best_hyperpar_df = pd.DataFrame(
        {k: [v] for k, v in best_hyperpars.items()})
    best_hyperpar_df.to_csv(OUTPUT_FILE, index=False)