#!/usr/bin/env python3
## TO-DO:
# - make project directory dynamic
# - put MNIST dataloaders in separate script
from pathlib import Path
PROJECT_DIR = Path("/home/alzub/projects/ggdl_self_learn")


if __name__ == "__main__":
    import argparse
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from tqdm import tqdm
    from utilities.prebuilt_nns import create_mnist_classifier

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-o", "--output-file", type=str,
        help="where to store the final model file in"
    )
    common_parser.add_argument(
        "-e", "--epochs", type=int, default=100,
        help="number of epochs to train the model for (default: 100)"
    )
    common_parser.add_argument(
        "-b", "--batch-size", type=int, default=256,
        help="batch size fpr dataloader (default: 256)"
    )
    common_parser.add_argument(
        "-lr", "--learning-rate", type=float, default=0.001,
        help="learning rate during training (default: 0.001)"
    )
    common_parser.add_argument(
        "-s", "--random-seed", type=int, default=1,
        help="random seed used for Python, NumPy and PyTorch (default: 1)"
    )

    parser = argparse.ArgumentParser(
        prog="mnist_classification.py",
        description="train a simple MNIST classifier",
        parents=[common_parser]
    )

    cli_args = parser.parse_args()

    OUTPUT_FILE = Path(cli_args.output_file)
    EPOCHS = cli_args.epochs
    BATCH_SIZE = cli_args.batch_size
    LR = cli_args.learning_rate
    SEED = cli_args.random_seed

    random.seed(SEED)
    torch.manual_seed(SEED)


    mnist_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            PROJECT_DIR / "data", train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            PROJECT_DIR / "data", train=False, download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        ),
        batch_size=BATCH_SIZE, shuffle=True
    )

    model = create_mnist_classifier().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fun = nn.CrossEntropyLoss()

    model.train()
    total_steps = len(mnist_train_loader)*EPOCHS
    progress_bar = tqdm(range(total_steps), desc="Training")
    for epoch in range(EPOCHS):
        for x, y in mnist_train_loader:
            optimizer.zero_grad()

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            y_hat_logits = model(x, return_final_feats=False)
            loss = loss_fun(y_hat_logits, y)

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(
                loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{EPOCHS}")
            progress_bar.update()
    
    torch.save(model.state_dict(), OUTPUT_FILE)