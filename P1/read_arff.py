from scipy.io import arff
import pandas as pd
import numpy as np
import sys
import os

def parse_to_csv(in_file):
    """
    Parses an .arff input file to the .csv format

    param in_file: Input .arff file to convert
    """

    data, meta = arff.loadarff(in_file)
    parsed_df = pd.DataFrame(data)

    if 'ionosphere' in in_file:
        parsed_df['class'] = np.where(parsed_df['class'] == b'g', 0, 1)
    else:
        parsed_df['class'] = parsed_df['class'].astype(np.int)

    out_file = in_file.split('.')[0] + '.csv'

    parsed_df.to_csv(out_file)


def main():
    # Get input file
    in_file = sys.argv[1]

    if not os.path.isfile(in_file):
        raise ValueError('Input must be a file')

    if not in_file.endswith('.arff'):
        raise ValueError('Input file must end in .arff')

    # Parse the file to .csv format
    parse_to_csv(in_file)


if __name__ == "__main__":
    main()
