import os
import argparse
import sys
sys.path.append('./networks')
import mnist_classifier
import mnist_gan


def main():
    parser = argparse.ArgumentParser(description="Arguments for running \
                            example networks in pytorch")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train over (default: 50)')
    parser.add_argument('--phase', required=True, default='train',
                        help='Set the phase of the network (default: train)')
    parser.add_argument('--dataset_list', required=True,
                        help='Path to csv containing the dataset and labels')
    parser.add_argument('--network', required=True,
                        help='The network which want to run: gan, classifier')
    args = parser.parse_args()

    if args.network == 'gan':
        networks.mnist_gan.main(args)
    elif args.network == 'classifier':
        networks.mnist_classifier.main(args)


if __name__ == '__main__':
    main()