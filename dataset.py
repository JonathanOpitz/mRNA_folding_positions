import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import os
import random


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

base_dir = os.path.dirname(__file__)
temp_csv = open(f"{base_dir}/data/elife-45396-fig1-data2-v2.csv", mode="r").readlines()
codon_opt = []
codon_neg = []
for line in temp_csv:
    line = line.strip().split(",")
    if line[0] == "codon":
        continue
    temp_codon = line[0]
    temp_flag1 = float(line[5])
    temp_flag2 = float(line[6])
    if temp_flag1 < 0 or temp_flag2 < 0:
        codon_neg.append(temp_codon)
    else:
        codon_opt.append(temp_codon)

codon_group_opt = []
codon_group_neg = []

for codon_list in codon_group:
    temp_codon_list_opt = []
    temp_codon_list_neg = []
    for temp_codon in codon_list:
        if temp_codon in codon_opt:
            temp_codon_list_opt.append(temp_codon)
        else:
            temp_codon_list_neg.append(temp_codon)
    codon_group_opt.append(temp_codon_list_opt)
    codon_group_neg.append(temp_codon_list_neg)


my_vocab = {
    "NNN": 64,
    "TTC": 1,
    "TTA": 2,
    "TTG": 3,
    "CTT": 4,
    "CTC": 5,
    "CTA": 6,
    "CTG": 7,
    "ATT": 8,
    "ATC": 9,
    "ATA": 10,
    "ATG": 11,
    "GTT": 12,
    "GTC": 13,
    "GTA": 14,
    "GTG": 15,
    "TAT": 16,
    "TAC": 17,
    "TAA": 18,
    "TAG": 19,
    "CAT": 20,
    "CAC": 21,
    "CAA": 22,
    "CAG": 23,
    "AAT": 24,
    "AAC": 25,
    "AAA": 26,
    "AAG": 27,
    "GAT": 28,
    "GAC": 29,
    "GAA": 30,
    "GAG": 31,
    "TCT": 32,
    "TCC": 33,
    "TCA": 34,
    "TCG": 35,
    "CCT": 36,
    "CCC": 37,
    "CCA": 38,
    "CCG": 39,
    "ACT": 40,
    "ACC": 41,
    "ACA": 42,
    "ACG": 43,
    "GCT": 44,
    "GCC": 45,
    "GCA": 46,
    "GCG": 47,
    "TGT": 48,
    "TGC": 49,
    "TGA": 50,
    "TGG": 51,
    "CGT": 52,
    "CGC": 53,
    "CGA": 54,
    "CGG": 55,
    "AGT": 56,
    "AGC": 57,
    "AGA": 58,
    "AGG": 59,
    "GGT": 60,
    "GGC": 61,
    "GGA": 62,
    "GGG": 63,
    "TTT": 0,
}

dict_vocab = {}
dict_vocab_inv = {}
for k, v in my_vocab.items():
    dict_vocab[k] = v
    dict_vocab_inv[v] = k

codon_group_id = []
for codons in codon_group:
    temp_codon_ids = []
    for codon in codons:
        temp_codon_ids.append(dict_vocab[codon])
    codon_group_id.append(temp_codon_ids)

dict_codon_group = {}
for codon_ids in codon_group_id:
    for temp_id in codon_ids:
        dict_codon_group[temp_id] = codon_ids

codon_group_id = []
for codons in codon_group_opt:
    temp_codon_ids = []
    for codon in codons:
        temp_codon_ids.append(dict_vocab[codon])
    codon_group_id.append(temp_codon_ids)

dict_codon_group_opt = {}
for codon_ids in codon_group_id:
    for temp_id in codon_ids:
        dict_codon_group_opt[temp_id] = codon_ids

codon_group_id = []
for codons in codon_group_neg:
    temp_codon_ids = []
    for codon in codons:
        temp_codon_ids.append(dict_vocab[codon])
    codon_group_id.append(temp_codon_ids)

dict_codon_group_neg = {}
for codon_ids in codon_group_id:
    for temp_id in codon_ids:
        dict_codon_group_neg[temp_id] = codon_ids


def read_lines(lines, max_len):
    res_k = []
    res_s = []
    for line in lines:
        line = line.strip("\n")
        line = line.replace("U", "T")
        try:
            if len(line) % 3 != 0:
                temp_len = len(line) // 3
                line = line[: temp_len * 3]
            line, line_s = process_line(line, max_len)
        except Exception as e:
            print(e)
            print(line)
            continue
        res_k.append(line)
        res_s.append(line_s)
    return res_k, res_s


def read_data(filename, max_len):
    with open(filename, mode="r") as f:
        lines = f.readlines()

    return read_lines(lines, max_len)


def process_line(line, max_len):
    res = []
    res_s = ""
    for i in range(0, len(line), 3):
        codon = line[i : i + 3]
        res.append(dict_vocab[codon])
        res_s += codon
    if len(res) >= max_len // 3:
        res = res[: max_len // 3]
        res_s = res_s[:max_len]
    else:
        for _ in range((max_len // 3) - len(res)):
            res.append(dict_vocab["NNN"])
            res_s += "NNN"
    return res, res_s


def read_sequence_and_encode(file_path, codon_table):
    with open(file_path, "r") as file:
        sequence = file.readline().strip()
    encoding = []
    for i in range(0, len(sequence), 3):
        codon = sequence[i : i + 3]
        one_hot = [0] * 65
        if codon in codon_table:
            index = codon_table[codon]
            one_hot[index] = 1
        encoding.append(one_hot)
    return [encoding]


def codon_convert_random(seq, ratio=0.5):
    seq_con = []
    status_seq = []
    for id_seq in seq:
        if random.random() < ratio:
            id_seq_new = random.choice(dict_codon_group[id_seq])
            seq_con.append(id_seq_new)
            if id_seq_new != id_seq:
                status_seq.append(1)
            else:
                status_seq.append(0)
        else:
            seq_con.append(id_seq)
            status_seq.append(0)

    return seq_con, status_seq


def codon_convert_random_auto(seq, style="all"):
    seq_con = []
    status_seq = []
    status_seq_mask = []

    for id_seq in seq:
        temp_ratio = random.choice([0.01, 0.05, 0.1])
        if random.random() < temp_ratio:
            if style == "all":
                id_seq_new = random.choice(dict_codon_group[id_seq])
            elif style == "opt":
                try:
                    id_seq_new = random.choice(dict_codon_group_opt[id_seq])
                except Exception as e:
                    id_seq_new = id_seq
            else:
                try:
                    id_seq_new = random.choice(dict_codon_group_neg[id_seq])
                except Exception as e:
                    id_seq_new = id_seq
            seq_con.append(id_seq_new)
            status_seq.append(random.random() / 2 + 0.50)
            status_seq_mask.append(1)
        else:
            seq_con.append(id_seq)
            status_seq.append(random.random() / 2.0)
            status_seq_mask.append(0)

    return seq_con, status_seq, status_seq_mask


def get_ids_from_dist(temp_dist):
    ids = []
    for id, prob in enumerate(temp_dist):
        if prob >= 0.95:
            ids.append(id)
            break
        elif prob >= 0.1:
            ids.append(id)
        else:
            continue
    return ids


def codon_convert_random_dist(seq, dist=None):
    seq_con = []
    status_seq = []
    status_seq_mask = []

    for id, id_seq in enumerate(seq):
        temp_dist = dist[id]
        try:
            id_seq_new = random.choice(get_ids_from_dist(temp_dist))
        except Exception as e:
            id_seq_new = id_seq
        seq_con.append(id_seq_new)
        status_seq.append(random.random() / 2.0)
        status_seq_mask.append(0)
    return seq_con, status_seq, status_seq_mask


def codon_convert_random_test(seq):
    seq_con = []
    status_seq = []

    for id_seq in seq:
        id_seq_new = random.choice(dict_codon_group[id_seq])
        seq_con.append(id_seq_new)
        if id_seq_new != id_seq:
            status_seq.append(random.random() / 2 + 0.50)
        else:
            status_seq.append(random.random() / 2.0)

    return seq_con, status_seq


dict_id2onehot = {}
for i in range(len(my_vocab)):
    temp_onehot = [0] * len(my_vocab)
    temp_onehot[i] = 1
    dict_id2onehot[i] = np.array(temp_onehot)


def convert_onehot(seq):
    seq_onehot = []
    for id_seq in seq:
        seq_onehot.append(dict_id2onehot[id_seq])
    return seq_onehot


def convert_bp(seq):
    seq_str = ""
    for id_seq in seq:
        seq_str += dict_vocab_inv[id_seq]
    return seq_str


class Dataset_rna_trans(Dataset):
    def __init__(
        self, model_config, dataset_name="train", status="train", data_scale=100, seq=None
    ):
        self.dataset_name = dataset_name
        self.model_config = model_config
        folder = model_config["data_folder"]
        max_len = model_config["len_gene_ori"]
        filename = os.path.join(folder, dataset_name + "_data.txt")

        if dataset_name == "train":
            self.seqs, self.seqs_s = read_lines([seq], max_len)
        else:
            self.seqs, self.seqs_s = read_data(filename, max_len)
        self.seqs = self.seqs * (data_scale // len(self.seqs))
        self.seqs_s = self.seqs_s * (data_scale // len(self.seqs))
        self.status = status
        self.codon_style = model_config["codon_style"]

    def __getitem__(self, id):
        seq = self.seqs[id]
        if self.status == "train":
            seq_con, status_mask, status_label_mask = codon_convert_random_auto(
                seq.copy(), self.codon_style
            )
            seq_tensor = torch.from_numpy(np.array(seq)).long()
            seq_con_tensor = torch.from_numpy(np.array(seq_con)).long()
            mask_tensor = torch.from_numpy(np.array(status_mask)).float()
            label_mask_tensor = torch.from_numpy(np.array(status_label_mask)).float()
            data = {}
            data["seq_ori"] = seq_tensor
            data["seq_con"] = seq_con_tensor
            data["mask_seq"] = mask_tensor
            data["mask_label"] = label_mask_tensor
            return data
        else:
            seq_tensor = torch.from_numpy(np.array(seq)).float()
            data = {}
            data["seq_con"] = seq_tensor
            return data

    def __len__(self):
        return len(self.seqs)


class Dataset_rna(Dataset):
    def __init__(
        self,
        model_config,
        dataset_name="train",
        status="train",
        data_scale=100,
        dist=None,
        seq=None,
    ):
        self.dataset_name = dataset_name
        self.model_config = model_config
        folder = model_config["data_folder"]
        max_len = model_config["len_gene_ori"]
        filename = os.path.join(folder, dataset_name + "_data.txt")
        if dataset_name == "train":
            self.seqs, self.seqs_s = read_lines([seq], max_len)
        else:
            self.seqs, self.seqs_s = read_data(filename, max_len)
        self.seqs = self.seqs * (data_scale // len(self.seqs))
        self.seqs_s = self.seqs_s * (data_scale // len(self.seqs))
        self.status = status
        self.codon_style = model_config["codon_style"]
        self.sampler_type = model_config["sampler_type"]
        self.dist = dist

    def __getitem__(self, id):
        seq = self.seqs[id]

        if self.status == "train":
            if "dist" in self.sampler_type and self.dist != None:
                if random.random() < 0.7:
                    seq_con, status_mask, status_label_mask = codon_convert_random_dist(
                        seq, self.dist
                    )
                else:
                    seq_con, status_mask, status_label_mask = codon_convert_random_auto(
                        seq, self.codon_style
                    )
            else:
                seq_con, status_mask, status_label_mask = codon_convert_random_auto(
                    seq, self.codon_style
                )
            seq = convert_onehot(seq)
            seq_con = convert_onehot(seq_con)
            seq_tensor = torch.from_numpy(np.array(seq)).float()
            seq_con_tensor = torch.from_numpy(np.array(seq_con)).float()
            mask_tensor = torch.from_numpy(np.array(status_mask)).float()
            label_mask_tensor = torch.from_numpy(np.array(status_label_mask)).float()
            data = {}
            data["seq_ori"] = seq_tensor.permute(1, 0)
            data["seq_con"] = seq_con_tensor.permute(1, 0)
            data["mask_seq"] = mask_tensor
            data["mask_label"] = label_mask_tensor
            return data
        else:
            seq_tensor = torch.from_numpy(np.array(seq)).float()
            data = {}
            data["seq_con"] = seq_tensor
            return data

    def __len__(self):
        return len(self.seqs)


class Dataset_rna_mfe(Dataset):
    def __init__(self, model_config, list_data, status="train", data_scale=1000000):
        self.model_config = model_config
        self.ori_len = model_config["len_gene_ori"]

        self.seqs = []
        self.mfe = []
        for seq, mfe in list_data:
            self.mfe.append(float(mfe))
            self.seqs.append(process_line(seq, self.ori_len)[0])

        temp_scale = data_scale // len(self.seqs)
        if temp_scale < 1:
            temp_scale = 1
        self.seqs = self.seqs * temp_scale
        self.mfe = self.mfe * temp_scale
        self.status = status

    def __getitem__(self, id):
        seq = self.seqs[id]
        seq = convert_onehot(seq)
        mfe = self.mfe[id]

        if self.status == "train":
            pass
        seq_tensor = torch.from_numpy(np.array(seq)).float()
        mfe_tensor = torch.from_numpy(np.array(mfe)).float()
        data = {}
        data["seq"] = seq_tensor.permute(1, 0)
        data["mfe"] = mfe_tensor

        return data

    def __len__(self):
        return len(self.seqs)
