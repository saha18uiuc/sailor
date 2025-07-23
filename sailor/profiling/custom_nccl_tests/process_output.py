import pandas as pd
import argparse

def process_data(input_file, world_size):
    tensor_size_bytes = []
    bandwidth = []

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Bandwidth' in line:
                tokens = line.split(",")
                size = tokens[0].split(" ")[-1]
                bw = tokens[1].split(" ")[-2]
                tensor_size_bytes.append(int(size))
                bandwidth.append(float(bw))

    results_df = pd.DataFrame(data={"bytes": tensor_size_bytes, "bw_GB_s": bandwidth})
    output_file = f"nccl_data_{world_size}.csv"
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, required=True, help='World size')
    parser.add_argument('--input_file', type=str, required=True, help='File containing nccl measurements')
    args = parser.parse_args()
    process_data(args.input_file, args.world_size)
