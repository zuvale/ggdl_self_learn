#!/usr/bin/env python3
# TO-DO:
# - put MNIST dataloaders in separate script
from pathlib import Path
import sys
from typing import List

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
    from torchvision import datasets, transforms
    from hpo.basic_nn import UNET_BOUNDS, OPTIM_BOUNDS
    from hpo.gen_model import (
        DIFFUSION_BOUNDS, VDM_DIFFUSION_BOUNDS,
        DDPMUNetMNISTSearchProblem, VDMUNetMNISTSearchProblem
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
        "-t", "--no-of-timesteps", type=int, default=1000,
        help="number of (de)noising timesteps (default: 1000)"
    )
    common_parser.add_argument(
        "-s", "--random-seed", type=int, default=1,
        help="random seed used for Python, NumPy and PyTorch (default: 1)"
    )
    common_parser.add_argument(
        "--warm-start-file", type=str,
        help="CSV file with a previous best solution used to seed the first population member"
    )
    parser = argparse.ArgumentParser(
        prog="diffusion_unet_nas.py",
        description="perform a naive neural architecture search of a diffusion model configuration using metaheuristic optimization",
        parents=[common_parser]
    )

    subparsers = parser.add_subparsers(dest="noising_style", required=True)

    ddpm_parent = argparse.ArgumentParser(add_help=False)
    ddpm_parser = subparsers.add_parser(
        name="ddpm",
        prog="diffusion_unet_nas.py ddpm",
        description="use a DDPM forward noising schedule (i.e., constant linear)",
        parents=[common_parser, ddpm_parent]
    )

    vdm_parent = argparse.ArgumentParser(add_help=False)
    vdm_parser = subparsers.add_parser(
        name="vdm",
        prog="diffusion_unet_nas.py vdm",
        description="use a VDM forward noising schedule (i.e., learnable linear)",
        parents=[common_parser, ddpm_parent]
    )

    cli_args = parser.parse_args()

    OUTPUT_FILE = Path(cli_args.output_file)
    GENERATIONS = cli_args.generations
    POP_SIZE = cli_args.population_size
    EPOCHS = cli_args.epochs
    BATCH_SIZE = cli_args.batch_size
    TIMESTEPS = cli_args.no_of_timesteps
    SEED = cli_args.random_seed
    WARM_START_FILE = (
        None
        if cli_args.warm_start_file is None
        else Path(cli_args.warm_start_file)
    )

    random.seed(SEED)
    torch.manual_seed(SEED)

    def build_starting_solutions(
        problem, pop_size: int, warm_start_file: Path
    ) -> List:
        import numpy as np


        if not warm_start_file.is_file():
            raise FileNotFoundError(
                f"Warm-start file not found: {warm_start_file}"
            )

        prior_best = pd.read_csv(warm_start_file)
        prior_best = prior_best.loc[
            :, ~prior_best.columns.astype(str).str.startswith("Unnamed:")
        ]
        prior_best = prior_best.iloc[0].to_dict()

        # Start from a valid random candidate so any VDM-only fields that are
        # missing from a DDPM run still get sensible values.
        seeded_candidate = {
            bound.name: value
            for bound, value in zip(
                problem.bounds, problem.generate_solution(encoded=False)
            )
        }
        seeded_candidate.update({
            bound.name: prior_best[bound.name]
            for bound in problem.bounds if bound.name in prior_best
        })

        ordered_candidate = [
            v if isinstance(v, (list, tuple, np.ndarray)) else [v]
            for v in (seeded_candidate[bound.name] for bound in problem.bounds)
        ]
        seeded_solution = problem.correct_solution(
            problem.encode_solution(ordered_candidate)
        )
        return [seeded_solution] + [
            problem.generate_solution() for _ in range(pop_size - 1)
        ]


    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            PROJECT_DIR / "data", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
                transforms.Lambda(lambda x: (x-0.5)*2.0),
            ])
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            PROJECT_DIR / "data", train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=BATCH_SIZE*4, shuffle=True
    )

    if cli_args.noising_style == "ddpm":
        problem = DDPMUNetMNISTSearchProblem(
            bounds=UNET_BOUNDS + OPTIM_BOUNDS + DIFFUSION_BOUNDS,
            data_shape=(1, 28, 28), train_loader=mnist_train_loader,
            test_loader=mnist_test_loader, n_epochs=EPOCHS,
            n_timesteps=TIMESTEPS, device=DEVICE
        )
    elif cli_args.noising_style == "vdm":
        problem = VDMUNetMNISTSearchProblem(
            bounds=UNET_BOUNDS + OPTIM_BOUNDS + VDM_DIFFUSION_BOUNDS,
            data_shape=(1, 28, 28), train_loader=mnist_train_loader,
            test_loader=mnist_test_loader, n_epochs=EPOCHS,
            n_timesteps=TIMESTEPS, device=DEVICE
        )
    
    optimizer_hpo = L_SHADE(epoch=GENERATIONS, pop_size=POP_SIZE)
    starting_solutions = None
    if WARM_START_FILE is not None:
        problem.set_seed(SEED)
        starting_solutions = build_starting_solutions(
            problem, POP_SIZE, WARM_START_FILE)

    optimizer_hpo.solve(
        problem, seed=SEED, starting_solutions=starting_solutions)

    best_hyperpars = optimizer_hpo.problem.decode_solution(
        optimizer_hpo.g_best.solution)
    print(best_hyperpars)
    best_hyperpar_df = pd.DataFrame(
        {k: [v] for k, v in best_hyperpars.items()})
    best_hyperpar_df.to_csv(OUTPUT_FILE, index=False)
