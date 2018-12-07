import os
import argparse



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
                        help='The network which want to run')
    parser.parse_args()




if __name__ == '__main__':
    main()