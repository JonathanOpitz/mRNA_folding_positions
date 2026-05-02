import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cds", type=str, required=True, help="your CDS name")
    parser.add_argument("--cds_seq", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--beta", type=float, default=100)
    parser.add_argument("--mfe_weight", type=float, default=0.7, help="float in [0, 1]")
    parser.add_argument(
        "--env",
        type=str,
        default="HEK293T",
        help="choose one from ['HEK293T', 'BJ', 'A549', 'HeLa'], or your custom env name",
    )
    parser.add_argument("--optim_epoch", type=int, default=20, help="optim epoch num")
    parser.add_argument("--csv", type=str, help="your custom env csv file")

    args = parser.parse_args()

    if args.mfe_weight < 0 or args.mfe_weight > 1:
        print(f"--mfe_weight must in [0, 1], you are using {args.mfe_weight}")
        exit(-1)

    base_dir = os.path.dirname(__file__)
    run_code = os.path.join(base_dir, "train_optim_sim.py")
    command = [
        "python3",
        run_code,
        "--cds",
        args.cds,
        "--cds_seq",
        args.cds_seq,
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--mfe_weight",
        str(args.mfe_weight),
        "--env",
        args.env,
        "--optim_epoch",
        str(args.optim_epoch),
    ]
    if args.csv:
        command = command + ["--csv", str(args.csv)]

    result = subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
