import argparse

from game import MineSweeperEnv
from dqn import MineSweeperTrainer
from models import MineSweeperCNN


FILE_PREFIX = "test"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--board_rows", type=int)
    parser.add_argument("--board_cols", type=int)
    parser.add_argument("--n_mines", type=int)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--flat_action", action="store_true")
    parser.add_argument("--n_episodes", type=int)
    parser.add_argument("--print_board", action="store_true")

    parser.add_argument("--n_channels", type=int)
    parser.add_argument("--model_depth", type=int)

    args = parser.parse_args()

    # Prepare game environment
    print("Preparing game environment...")
    env = MineSweeperEnv(
        board_size=(args.board_rows, args.board_cols),
        n_mines=args.n_mines,
        flat_action=args.flat_action,
    )

    # Prepare models
    print("Preparing models...")
    policy_net = MineSweeperCNN(
        board_size=(args.board_rows, args.board_cols),
        n_channels=args.n_channels,
        depth=args.model_depth,
    )
    target_net = MineSweeperCNN(
        board_size=(args.board_rows, args.board_cols),
        n_channels=args.n_channels,
        depth=args.model_depth,
    )

    # Define trainer
    print("Preparing trainer...")
    trainer = MineSweeperTrainer(batch_size=args.batch_size)
    trainer.register(
        policy_net=policy_net,
        target_net=target_net,
        env=env,
    )

    # Train
    print("Start training")
    trainer.train(n_episodes=args.n_episodes, print_board=args.print_board)

    # Plot result
    print("Training complete")
    trainer.plot_result(FILE_PREFIX)


if __name__ == "__main__":
    main()
