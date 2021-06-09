import argparse
import os
import re
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default="./model_checkpoints", type=str)
    parser.add_argument("--prefix", default="", type=str)
    parser.add_argument("--keep_latest", default=1, type=int)
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    num_keep = args.keep_latest

    assert prefix != "", "Please specify a proper prefix"

    target_prefix = set()
    for f in os.listdir(directory):
        if f.startswith(prefix):
            target_prefix.add(re.sub(r"(\_\d+)+$", "", f))

    if not len(target_prefix):
        print("No file to remove")
        sys.exit()

    remove_list = []
    
    for prefix in target_prefix:
        files_with_prefix = [f for f in os.listdir(directory) if re.sub(r"(\_\d+)+$", "", f) == prefix]
        files_with_prefix = sorted(files_with_prefix, key=lambda s: s.split("_")[-1])
        files_to_remove = files_with_prefix[:-num_keep]
        remove_list.append(files_to_remove)
        print(f"Pending task: {prefix}, {files_to_remove[0].split('_')[-1]} ~ {files_to_remove[-1].split('_')[-1]} ({len(files_to_remove)} files)")

    instruction = input("Continue? [y/n]")
    if instruction.lower() == "y":
        for f in sum(remove_list, []):
            os.remove(os.path.join(directory, f))
        print("Done!")
    else:
        print("Aborted.")