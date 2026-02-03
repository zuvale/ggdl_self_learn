#!/usr/bin/env python3
# TO-DO:
# - make project directory dynamic
# - put MNIST dataloaders in separate script
from pathlib import Path
PROJECT_DIR = Path("/home/alzub/projects/ggdl_self_learn")


if __name__ == "__main__":
    import argparse
    from mealpy.evolutionary_based.SHADE import L_SHADE
    import pandas as pd
    import random
    import torch
    from torchvision import datasets, transforms
    from hpo.basic_nn import UNET_BOUNDS, OPTIM_BOUNDS
    from hpo.gen_model import DIFFUSION_BOUNDS, DDPMUNetMNISTSearchProblem

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
        help="batch size fpr dataloader (default: 256)"
    )
    common_parser.add_argument(
        "-t", "--no-of-timesteps", type=int, default=1000,
        help="number of (de)noising timesteps (default: 1000)"
    )
    common_parser.add_argument(
        "-s", "--random-seed", type=int, default=1,
        help="random seed used for Python, NumPy and PyTorch (default: 1)"
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

    random.seed(SEED)
    torch.manual_seed(SEED)


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

    if cli_args.noising_style == "ddpm":
        problem = DDPMUNetMNISTSearchProblem(
            bounds=UNET_BOUNDS + OPTIM_BOUNDS + DIFFUSION_BOUNDS,
            data_shape=(1, 28, 28), train_loader=mnist_train_loader,
            n_epochs=EPOCHS, n_timesteps=TIMESTEPS, device=DEVICE
        )
    
    optimizer_hpo = L_SHADE(epoch=GENERATIONS, pop_size=POP_SIZE)
    optimizer_hpo.solve(problem, seed=SEED)

    best_hyperpars = optimizer_hpo.problem.decode_solution(
        optimizer_hpo.g_best.solution)
    print(best_hyperpars)
    best_hyperpar_df = pd.DataFrame(
        {k: [v] for k, v in best_hyperpars.items()})
    best_hyperpar_df.to_csv(OUTPUT_FILE)