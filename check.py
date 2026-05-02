from typing import Optional
import numpy as np


codon_group = [
    ["TTT", "TTC"],
    ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    ["ATT", "ATC", "ATA"],
    ["ATG"],
    ["GTT", "GTC", "GTA", "GTG"],
    ["TAT", "TAC"],
    ["TAA", "TAG", "TGA"],
    ["CAT", "CAC"],
    ["CAA", "CAG"],
    ["AAT", "AAC"],
    ["AAA", "AAG"],
    ["GAT", "GAC"],
    ["GAA", "GAG"],
    ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    ["CCT", "CCC", "CCA", "CCG"],
    ["ACT", "ACC", "ACA", "ACG"],
    ["GCT", "GCC", "GCA", "GCG"],
    ["TGT", "TGC"],
    ["TGG"],
    ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    ["GGT", "GGC", "GGA", "GGG"],
    ["NNN"],
]


class Check:
    def __init__(self, orig_seq) -> None:
        # gene_name = model_config['data_folder'].split('/')[-2].split('_')[0]
        # data_dir = model_config["data_folder"] + "/train_data.txt"
        # with open(data_dir, mode="r") as f:
        #     lines = f.readlines()
        orig_seq = orig_seq.replace("U", "T")
        max_len = False
        orig_len = len(orig_seq)
        if max_len:
            if orig_len < max_len:
                for _ in range((max_len - orig_len) // 3):
                    self.orig_seq = orig_seq + "NNN"
            else:
                self.orig_seq = orig_seq[:max_len]
        else:
            self.orig_seq = orig_seq

    def search(self, codon: str) -> Optional[int]:
        for index, i in enumerate(codon_group):
            for j in i:
                if j == codon:
                    return index

        return None

    def analysis_error_codon(self, line):
        error_num = 0
        diff_num = 0
        for i in range(0, len(line), 3):
            codon = line[i : i + 3]
            orig_codon = self.orig_seq[i : i + 3]
            index_in_codon_group = self.search(orig_codon)
            if index_in_codon_group is None:
                break
            if codon != orig_codon:
                diff_num = diff_num + 1
            if codon not in codon_group[index_in_codon_group]:
                error_num = error_num + 1
        return error_num

    def calc_mean_error_rate(self, f_path: str, num=-1) -> float:
        with open(f_path, "r") as f:
            lines = f.readlines()
            lines = [i.split("\t")[0].strip().replace("U", "T") for i in lines if i]
        if num > 1:
            lines = lines[:num]

        all_error_num = []
        all_diff_num = []
        for index, line in enumerate(lines):
            error_num = 0
            diff_num = 0
            for i in range(0, len(line), 3):
                codon = line[i : i + 3]
                orig_codon = self.orig_seq[i : i + 3]
                index_in_codon_group = self.search(orig_codon)
                if index_in_codon_group is None:
                    break
                if codon != orig_codon:
                    diff_num = diff_num + 1
                if codon not in codon_group[index_in_codon_group]:
                    error_num = error_num + 1

            all_error_num.append(error_num)
            all_diff_num.append(diff_num)

        all_error_num = np.array(all_error_num)
        all_diff_num = np.array(all_diff_num)

        print("total num of diff sequences   :", len(set(lines)))
        print("AVG num of diff tokens per seq:", int(np.mean(all_diff_num)))
        print("AVG Error num of diff tokens  :", int(np.mean(all_error_num)))
        num_seqs_error = 0
        for n_e in all_error_num:
            if n_e > 0:
                num_seqs_error += 1
        print(
            "AVG error rate of seq level   : {:.3f}%".format(
                100 * num_seqs_error / len(all_error_num)
            )
        )
        temp_rate = float(np.mean(all_error_num) / (np.mean(all_diff_num) + 1e-6))
        print("AVG error rate of codon tokens: {:.3f}%".format(temp_rate * 100))
        return temp_rate, all_diff_num

    def post_process(self, f_path: str) -> list:
        with open(f_path, "r") as f:
            lines = f.readlines()
            lines = [i.strip() for i in lines if i]
        res = []
        for index, line in enumerate(lines):
            new_line = ""
            for i in range(0, len(line), 3):
                codon = line[i : i + 3]
                orig_codon = self.orig_seq[i : i + 3]
                index_in_codon_group = self.search(orig_codon)
                if codon not in codon_group[index_in_codon_group]:
                    new_line = new_line + orig_codon
                else:
                    new_line = new_line + codon
            res.append(new_line)

        return res


if __name__ == "__main__":
    pass
