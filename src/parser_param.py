import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="CNN Text Classification")

    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train [default: 256]')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training [default: 64]')
    
    # data 
    parser.add_argument('--num_words', type=int, default=2000, help='Number of words')
    parser.add_argument('--seq_len', type=int, default=35, help='Lenght of sequence')
    parser.add_argument('--embedding_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--out_size', type=int, default=32, help='Size of output')
    parser.add_argument('--stride', type=int, default=2, help='Number of stride')
    # device


    return parser.parse_args()