import json
import glob
import sys


def merge_profiles(file_list):
    merged_dict = {}
    for file in file_list:
        with open(file, 'r') as f:
            mbs_dict = json.load(f)
            for key, value in mbs_dict.items():
                if key in merged_dict:
                    merged_dict[key].update(value)
                else:
                    merged_dict[key] = value

    with open('profile.json', 'w') as f:
        json.dump(merged_dict, f)


if __name__ == "__main__":
    given_dir = sys.argv[1]
    file_list = [file for file in glob.glob(f"{given_dir}/profile_*")]
    merge_profiles(file_list)
