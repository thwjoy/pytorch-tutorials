import os
import argparse
import sys
import networks
sys.path.append('./torch_utils')


def main():
    parser = argparse.ArgumentParser(description="Arguments for running \
                            example networks in pytorch")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train over (default: 50)')
    parser.add_argument('--phase', required=False, default='train',
                        help='Set the phase of the network (default: train)')
    parser.add_argument('--dataset_list', required=False, default='mnist/training/list.txt'
                        help='Path to csv containing the dataset and labels')
    parser.add_argument('--network', required=True,
                        help='The network which want to run: gan, classifier')
    parser.add_argument('--log_message', required=False, default='',
                        help='A simple message to help debugging')
    args = parser.parse_args()

    if args.network == 'gan':
        networks.mnist_gan.train(args)
    elif args.network == 'cgan':
        networks.mnist_cgan.train(args)
    elif args.network == 'vae':
        networks.mnist_vae.train(args)
    elif args.network == 'classifier':
        networks.mnist_classifier.train(args)


if __name__ == '__main__':
    main()