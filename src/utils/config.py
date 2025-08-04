import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #### about train ####
    parser.add_argument('--deviceID', type=int, default=3)
    parser.add_argument('--chunk_size', type=int, default=50)
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--synapse', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default='VSENet')
    parser.add_argument('--patience', type=int, default=120)

    return parser.parse_args()