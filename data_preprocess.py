import math
import time

import numpy as np
import pandas as pd
import pyranges as pr
from Bio import SeqIO


def identity_positive_examples(tf):
    start_time = time.time()

    file_path = "Data/factorbookMotifPos.txt"
    with open(file_path, "r") as f:
        df = pd.read_csv(f, sep="\t", header=None)

    df.columns = ["Field1", "Chromosome", "Start", "End", "TF", "Score", "Strand"]
    df = df.drop(columns=["Field1", "Score"])  # Drop unnecessary columns
    df = df[df["Strand"] == "+"]  # Filter for the forward strand
    df_target = df[df["TF"] == tf]
    df_target = df_target[df_target["Chromosome"] != "chrY"]

    # Define the order of chromosomes as a categorical type
    list_chrs = ["chr%s" % s for s in range(1, 23)] + ["chrX"]
    df_target["Chromosome"] = pd.Categorical(
        df_target["Chromosome"], categories=list_chrs, ordered=True
    )

    # Sort the DataFrame by Chromosome in the defined order
    df_target = df_target.sort_values("Chromosome")
    df_target = df_target.drop(
        ["Strand"], axis=1
    )  # Drop the Strand column as it's no longer needed

    # Extract the genomic coordinates for the transcription factor
    positive_examples_target = df_target[["Chromosome", "Start", "End"]].values.tolist()

    print(
        "Length of Target TF binding site:",
        positive_examples_target[0][2] - positive_examples_target[0][1],
    )
    print("Length of Target TF Binding samples:", len(positive_examples_target))

    # Save the sorted DataFrame to a new file
    output_path = f"Data/Pos_Samples/factorbookMotifPos_{tf}.txt"
    df_target.to_csv(output_path, sep="\t", index=False, header=False)

    end_time = time.time()
    print("Running time: ", end_time - start_time)

    return output_path, positive_examples_target


def merge_fasta_file():
    chr_list = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    for chro in chr_list:
        print(chro)
        with open("Data/all_chr.fa", "a") as outfile:
            with open(f"Data/chromFa/{chro}.fa", "r") as infile:
                outfile.write(infile.read())


def read_fasta_file(fasta_file):
    sequences = {}
    with open(fasta_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            sequences[record.id] = str(record.seq)
    return sequences


def identify_possible_negative_samples(tf):
    # Create a dataframe from the textfile
    file_name = "Data/factorbookMotifPos.txt"  # Contains genomic coordinates of positive examples of the target TFs
    df = pd.read_csv(
        file_name,
        usecols=[1, 2, 3, 4, 5, 6],
        names=["Chom", "TFStart", "TFEnd", "TFName", "ScoreBS", "StrandBS"],
        header=None,
        sep="\t",
    )

    # Get an overview of the dataframe
    # Extract rows containing TBP binding sites information
    df_TF = df[df["TFName"] == tf]
    df_TF_fwd = df_TF[df_TF["StrandBS"] == "+"]
    # The bed file: contains active regulatory regions in GM12879 (a cell line derived from lymphoblasts)
    gr = pr.read_bed("Data/wgEncodeRegTfbsClusteredV3.GM12878.merged.bed")
    df_bed = gr.df

    # Observe the top and bottom of the bed file
    # List out all chromosomes
    list_choms = list(set(df_bed["Chromosome"].to_list()))
    """The list was created after inspecting the chromosomes contained in the bed file.
    The list of chromosomes were created as below for ordering (sorting the list of chromosomes generated directly from the bed
    file dataframe resulted in an undesirable ordering of chromosomes)"""
    list_chrs = ["chr%s" % s for s in range(1, 23)]
    list_chrs.append("chrX")
    ## Before the edit on df_TF_negative
    start_time_total = time.time()
    df_TF_negative_all = pd.DataFrame(
        columns=["Chom", "NStart", "NEnd", "UnboundTF"]
    )  # An empty dataframe used later

    for chom in list_chrs:  # For each chromosome (chr)
        start_time = time.time()
        print(chom)

        # Extract portion of the bed file corresponding to the chromosome
        df_bed_chom = df_bed[df_bed["Chromosome"] == chom]
        # print(df_bed_chom.head(10))
        # Extract portion of the genomic coordinate (of +'ve example) df corresponding to the chr
        df_TF_chom = df_TF_fwd[df_TF_fwd["Chom"] == chom]
        # print(df_TF_chom.head(10))

        # Sorting to ensure the dataframe is sorted
        df_bed_chom_s = df_bed_chom.sort_values("Start")
        df_TF_chom_s = df_TF_chom.sort_values("TFStart")

        # For reducing running time
        # Get range for the coordinates of active regulatory regions
        min_bed = df_bed_chom_s["Start"].min()
        max_bed = df_bed_chom_s["End"].max()

        # Get range for the coordinates of TF bound sites (positive examples)
        min_TF = df_TF_chom_s["TFStart"].min()
        max_TF = df_TF_chom_s["TFEnd"].max()

        """
        Idea:
        Negative examples will fall under the active regulatory region.
        => Potential negative examples will be regions in active regulatory regions that 
        doesn't loverlap with locations of positive examples
        i.e. Negative examples can't possibly be anywhere in the active regulatory region
        that overlaps with positive examples
        """
        # Extract all the coordinates of active regulatory region that comes before the
        # smallest coordinate among the positive examples for the chromosome
        df_bed_non_top = df_bed_chom_s[df_bed_chom_s["End"] < min_TF]
        df_bed_non_top = df_bed_non_top.rename(
            columns={"Chromosome": "Chom", "Start": "NStart", "End": "NEnd"}
        )
        if df_bed_non_top.empty == False:
            df_bed_non_top.loc[:, "UnboundTF"] = tf
        df_bed_non_top = df_bed_non_top.drop(["Name"], axis=1)

        # Extract all the coordinates of active regulatory region that comes after the
        # largest coordinate among the positive examples for the chromosome
        df_bed_non_bottom = df_bed_chom_s[
            max_TF < df_bed_chom_s["Start"]
        ]  # need to append these
        df_bed_non_bottom = df_bed_non_bottom.rename(
            columns={"Chromosome": "Chom", "Start": "NStart", "End": "NEnd"}
        )
        if df_bed_non_bottom.empty == False:
            df_bed_non_bottom.loc[:, "UnboundTF"] = tf
        df_bed_non_bottom = df_bed_non_bottom.drop(["Name"], axis=1)

        """
        min active      minTF                 maxTF            max active

        |=================|----------------------|===============|
        """
        # Extract coordinates of active regulatory region that may overlap with coordinates of positive examples
        df_bed_relev = df_bed_chom_s[df_bed_chom_s["Start"] <= max_TF]
        df_bed_relev = df_bed_relev[min_TF <= df_bed_relev["End"]]

        # Identify potential regative examples
        list_cols_TF = list(df_TF_chom_s.columns.values)
        df_TF_negative = pd.DataFrame(
            columns=["Chom", "NStart", "NEnd", "UnboundTF"]
        )  # Create an empty data frame

        count = 0
        for (
            index_b,
            row_b,
        ) in df_bed_relev.iterrows():  # For each active regulatory region (shortlisted)
            X = row_b["Start"]
            Y = row_b["End"]

            df_TF_Inregion = pd.DataFrame(
                columns=list_cols_TF
            )  # Create an empty dataframe
            # Gather any positive examples that may fall within the active regulatory region
            """
            X                           Y
            |======aaaa===aaaa====aaaa==|
            """
            for index_TF, row_TF in df_TF_chom_s.iterrows():
                start = row_TF["TFStart"]
                end = row_TF["TFEnd"]
                if (
                    X <= start and end <= Y
                ):  # If the coordinates of positive examples fall within the active regulatory region
                    df_1row = df_TF_chom_s.loc[[index_TF]]
                    df_TF_Inregion = df_TF_Inregion.append(df_1row)

            # If at least one positive example falls under the active regulatory region
            if df_TF_Inregion.empty == False:
                # print(count)
                df_TF_Inregion_r = df_TF_Inregion.reset_index()

                idx_count = df_TF_Inregion_r.shape[0]

                for index_InR, row_InR in df_TF_Inregion_r.iterrows():
                    st = row_InR["TFStart"]
                    ed = row_InR["TFEnd"]

                    if index_InR == 0:  # If it's the first row
                        new_row1 = {
                            "Chom": chom,
                            "NStart": X,
                            "NEnd": st - 1,
                            "UnboundTF": tf,
                        }
                        df_TF_negative = df_TF_negative.append(
                            new_row1, ignore_index=True
                        )

                    if index_InR == idx_count - 1:  # If it's the last row
                        new_row2 = {
                            "Chom": chom,
                            "NStart": ed + 1,
                            "NEnd": Y,
                            "UnboundTF": tf,
                        }
                        df_TF_negative = df_TF_negative.append(
                            new_row2, ignore_index=True
                        )
                        break
                    else:  # If there are multiple binding sites within a region and if the row is not the last row
                        st_2 = int(df_TF_Inregion_r.loc[[index_InR + 1]].TFStart)
                        ed_2 = int(df_TF_Inregion_r.loc[[index_InR + 1]].TFEnd)
                        if st_2 > ed:  # If the TF binding sites don't overlap
                            new_row = {
                                "Chom": chom,
                                "NStart": ed + 1,
                                "NEnd": st_2 - 1,
                                "UnboundTF": tf,
                            }
                            df_TF_negative = df_TF_negative.append(
                                new_row, ignore_index=True
                            )

            count += 1

            # Create a dataframe that contains all the negative examples
        all_negative_dfs = [df_bed_non_top, df_TF_negative, df_bed_non_bottom]
        df_negative_all = pd.concat(all_negative_dfs).reset_index(drop=True)
        end_time = time.time()
        run_time = end_time - start_time
        print("Running time for {}: {}".format(chom, run_time))

        frames = [df_TF_negative_all, df_negative_all]
        df_TF_negative_all = pd.concat(frames).reset_index(drop=True)

    end_time_total = time.time()
    run_time_total = end_time_total - start_time_total
    print(df_TF_negative_all.shape)
    print("Total running time: ", run_time_total)
    df_TF_negative_all.to_csv(
        f"Data/Neg_Samples/factorbookMotifNeg_{tf}.txt",
        sep="\t",
        index=False,
        header=False,
    )


def get_position_weight_matrix(tf):
    # Note: factorbookMotifPwm.txt contains PWM of all TFs that was listed in Part 1

    # Obtain PWM of the TF of interest
    with open("Data/factorbookMotifPwm.txt") as f:
        for line in f:
            if tf in line:  # If the line is for the TF
                line = line.split(",")
                for i in range(len(line)):
                    if i == 0:
                        tf_name = line[i].split("\t")[0]
                        tf_len = line[i].split("\t")[1]
                        tf_len = int(tf_len)
                        line[i] = line[i].split("\t")[-1]
                    else:
                        line[i] = line[i].replace("\t", "")
                del line[-1]
                print(tf_name)
                print(len(line))
                line_int = [float(i) for i in line]
                # Convert to a numpy array
                pwm_arr = np.array(line_int)
                pwm_arr.astype(int)
                pwm_arr2 = np.reshape(pwm_arr, (4, tf_len))
                print(pwm_arr2)
                print(
                    np.sum(pwm_arr2, axis=0)
                )  # Validating that the PWM makes sense (columns add up to 1)
                return pwm_arr2


def get_negative_samples(tf):
    pwm_arr2 = get_position_weight_matrix(tf)
    filename = "Data/Extracted_Samples/TBP_possible_neg_samples.fasta"
    list_chrs = ["chr%s" % s for s in range(1, 23)]
    list_chrs.append("chrX")
    fileout = f"Data/Extracted_Samples/{tf}_neg_samples.fasta"
    s_time = time.time()
    for chom in list_chrs:
        print("chom: ", chom)
        logp_thresh = -22

        with open(filename, "r") as fh:
            with open(fileout, "a") as fout:
                for seq_record in SeqIO.parse(fh, "fasta"):
                    if chom != seq_record.id[: seq_record.id.find(":")]:
                        continue
                    sequence = seq_record.seq
                    # print(sequence)
                    start_time = time.time()
                    # print("length of the seq:", len(sequence))
                    count = 0
                    for j in range(len(sequence) - 20):
                        # print("j:", j)
                        count += 1
                        k = 0
                        target_seq = sequence[j : j + 20]
                        if "N" in target_seq:
                            N_in = "YesN"
                        else:
                            N_in = "NoN"
                        if N_in == "NoN":
                            if (
                                target_seq[k + 12] != "G"
                                and target_seq[k + 12] != "g"
                                and target_seq[k + 13] != "G"
                                and target_seq[k + 13] != "g"
                                and target_seq[k + 14] != "G"
                                and target_seq[k + 14] != "g"
                                and target_seq[k + 14] != "C"
                                and target_seq[k + 14] != "c"
                                and target_seq[k + 15] != "G"
                                and target_seq[k + 15] != "g"
                                and target_seq[k + 15] != "C"
                                and target_seq[k + 15] != "c"
                                and target_seq[k + 16] != "G"
                                and target_seq[k + 16] != "g"
                                and target_seq[k + 17] != "G"
                                and target_seq[k + 17] != "g"
                            ):
                                # if target_seq[k+2] != "T" and target_seq[k+2] != "t" and target_seq[k+3] != "G" and target_seq[k+3] != "g" and target_seq[k+4] != "T" and target_seq[k+4] != "t" and target_seq[k+9] != "C" and target_seq[k+9] != "c" and target_seq[k+9] != "A" and target_seq[k+9] != "a":
                                log_prob = 0
                                for i in range(20):
                                    if target_seq[i] == "A" or target_seq[i] == "a":
                                        arr_idx = 0
                                    elif target_seq[i] == "C" or target_seq[i] == "c":
                                        arr_idx = 1
                                    elif target_seq[i] == "G" or target_seq[i] == "g":
                                        arr_idx = 2
                                    elif target_seq[i] == "T" or target_seq[i] == "t":
                                        arr_idx = 3
                                    else:  # if "N"
                                        break

                                    # print(pwm_arr2[arr_idx, i])
                                    log_prob += math.log(pwm_arr2[arr_idx, i])
                                # print("log_prob:", log_prob)
                                if log_prob > logp_thresh:  # if sum_logprob > -22
                                    # print("Written")
                                    # print(target_seq)
                                    # print(type(str(target_seq)))
                                    fout.write(
                                        f">{seq_record.id}_{j}-{j+20}\n{target_seq}\n"
                                    )
                                    # SeqIO.write(target_seq, fout, "fasta")
                                    # count2 += 1
                                    # print(count2)

                    # print("count:", count)
                    end_time = time.time()
                    time_taken = end_time - start_time
                    # print("run time:", time_taken)
    e_time = time.time()
    r_time = e_time - s_time
    print("Running time of grid search to identify negative examples:", r_time)


if __name__ == "__main__":
    # Merge fasta files
    # merge_fasta_file()
    # identity_positive_examples(tf="TBP")
    # identify_possible_negative_samples(tf="TBP")
    # get_position_weight_matrix(tf="TBP")
    get_negative_samples(tf="TBP")
