import math

from Bio import SeqIO

from data_preprocess import get_position_weight_matrix


def calculate_negative_samples_log_prob_threshold(tf):
    pwm_arr2 = get_position_weight_matrix(tf)
    filename = "Data/Extracted_Samples/TBP_pos_samples.fasta"
    list_chrs = ["chr%s" % s for s in range(1, 23)]
    list_chrs.append("chrX")
    max_logprob_all = 0
    for chom in list_chrs:
        print("chom: ", chom)
        sum_logprob = 0
        avg_logprob = 0
        max_logprob = 0

        n_seqs = [
            rec.id
            for rec in SeqIO.parse(filename, "fasta")
            if chom == rec.id[: rec.id.find(":")]
        ]
        print("# of seqs: ", len(n_seqs))

        with open(filename, "r") as fh:
            count = 0
            for seq_record in SeqIO.parse(fh, "fasta"):
                if chom != seq_record.id[: seq_record.id.find(":")]:
                    continue
                sequence = seq_record.seq
                target_seq = sequence
                k = 0
                log_prob = 0
                if (
                    target_seq[k + 2] != "T"
                    and target_seq[k + 2] != "t"
                    and target_seq[k + 3] != "G"
                    and target_seq[k + 3] != "g"
                    and target_seq[k + 4] != "T"
                    and target_seq[k + 4] != "t"
                    and target_seq[k + 9] != "C"
                    and target_seq[k + 9] != "c"
                    and target_seq[k + 9] != "A"
                    and target_seq[k + 9] != "a"
                ):
                    # print("Success")
                    count += 1
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
                        log_prob += math.log(pwm_arr2[arr_idx, i])
                    sum_logprob += log_prob
                    if max_logprob > log_prob:
                        max_logprob = log_prob

            avg_logprob = sum_logprob / len(n_seqs)
            print("count: ", count)
            print("avg_logprob: ", avg_logprob)
            print("max_logprob: ", max_logprob)
            print("=============")
        if max_logprob_all > max_logprob:
            max_logprob_all = max_logprob
        print("max_logprob_all: ", max_logprob_all)
        logp_thresh = math.floor(max_logprob_all)

        print("logp_thresh: ", logp_thresh)


if __name__ == "__main__":
    calculate_negative_samples_log_prob_threshold(tf="TBP")
