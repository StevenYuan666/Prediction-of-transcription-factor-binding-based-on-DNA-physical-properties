import os

import pandas as pd
from Bio import SeqIO

# Directory containing your data files
data_directory = "Data"

# File paths
genome_dir = os.path.join(data_directory, "chromFa")
regulatory_regions_file = os.path.join(
    data_directory, "wgEncodeRegTfbsClusteredV3.GM12878.merged.bed"
)
tf_binding_sites_file = os.path.join(data_directory, "factorbookMotifPos.txt")

# Transcription factor of interest
your_transcription_factor = "CTCF"

FASTA_file_list = [
    "chr1.fa",
    "chr2.fa",
    "chr3.fa",
    "chr4.fa",
    "chr5.fa",
    "chr6.fa",
    "chr7.fa",
    "chr8.fa",
    "chr9.fa",
    "chr10.fa",
    "chr11.fa",
    "chr12.fa",
    "chr13.fa",
    "chr14.fa",
    "chr15.fa",
    "chr16.fa",
    "chr17.fa",
    "chr18.fa",
    "chr19.fa",
    "chr20.fa",
    "chr21.fa",
    "chr22.fa",
    "chrX.fa",
    "chrY.fa",
]


# Load all FASTA files into a dictionary
def load_genome_sequences(genome_dir):
    sequences = {}
    for filename in os.listdir(genome_dir):
        if filename in FASTA_file_list:
            path = os.path.join(genome_dir, filename)
            with open(path, "r") as fasta_file:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    sequences[record.id] = str(record.seq).upper()
    return sequences


# Load regulatory regions
def load_regulatory_regions(filepath):
    return pd.read_csv(
        filepath, sep="\t", header=None, names=["chrom", "start", "end", "name"]
    )


# Load transcription factor binding sites
def load_tf_binding_sites(filepath):
    return pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["ignore", "chrom", "start", "end", "tf_name", "score", "strand"],
    )


# Extract sequences based on coordinates
def extract_sequences(sequences, chrom, start, end, strand):
    seq = sequences[chrom][start:end]
    return seq[::-1].translate(str.maketrans("ACGT", "TGCA")) if strand == "-" else seq


# Find negative examples
def find_negative_examples(regulatory_regions, tf_sites, sequences):
    negative_examples = []
    for _, region in regulatory_regions.iterrows():
        # Check if the region overlaps with any TF binding site
        overlaps = tf_sites[
            (tf_sites["chrom"] == region["chrom"])
            & (tf_sites["start"] < region["end"])
            & (tf_sites["end"] > region["start"])
        ]
        if overlaps.empty:
            # If no overlap, this is a negative example
            seq = extract_sequences(
                sequences, region["chrom"], region["start"], region["end"], None
            )
            negative_examples.append(seq)
    return negative_examples


def print_unique_transcription_factors(tf_binding_sites):
    unique_tfs = tf_binding_sites["tf_name"].unique()
    print(f"There are {len(unique_tfs)} unique transcription factors.")
    return unique_tfs


# Main function to process the data
def process():
    # Load genomic sequences
    sequences = load_genome_sequences(genome_dir)

    # Load regulatory regions and TF binding sites
    regulatory_regions = load_regulatory_regions(regulatory_regions_file)
    tf_binding_sites = load_tf_binding_sites(tf_binding_sites_file)

    # Print unique transcription factors
    print_unique_transcription_factors(tf_binding_sites)

    # Filter for the specific TF
    tf_sites = tf_binding_sites[
        tf_binding_sites["tf_name"] == your_transcription_factor
    ]

    # Extract positive examples (sequences where the TF is known to bind)
    positive_examples = [
        extract_sequences(
            sequences, row["chrom"], row["start"], row["end"], row["strand"]
        )
        for _, row in tf_sites.iterrows()
    ]

    # Find negative examples
    negative_examples = find_negative_examples(regulatory_regions, tf_sites, sequences)

    # Now positive_examples and negative_examples contain the sequences for your ML model

    print("Positive examples: ", len(positive_examples))
    print("Negative examples: ", len(negative_examples))


if __name__ == "__main__":
    process()
