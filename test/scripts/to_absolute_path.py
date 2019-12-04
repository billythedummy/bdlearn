import argparse

parser = argparse.ArgumentParser(description="Add absolute path prefix to a train/test file")

parser.add_argument('out', help='output train/test file', type=str)
parser.add_argument('file', help='train/test file to change', type=str)
parser.add_argument('abs_prefix', help='absolute path prefix to add', type=str)

args = parser.parse_args()

with open(args.file, 'r') as f:
    with open(args.out, 'w+') as o:
        for line in f:
            line = args.abs_prefix.rstrip("/") + "/" + line.lstrip("/")
            o.write(line)
    