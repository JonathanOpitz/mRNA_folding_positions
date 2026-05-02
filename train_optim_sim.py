import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    import RNA
except Exception as e:
    print(e)
import json
import argparse
import random
import shutil
import time
from contextlib import suppress
from multiprocessing import Pool, cpu_count
from RiboDecode.dataset import (
    Dataset_rna,
    Dataset_rna_mfe,
    dict_vocab_inv,
    my_vocab,
    read_data,
    read_lines,
    process_line,
    dict_codon_group,
)
from RiboDecode.models import mfe_conv_sim
from RiboDecode.score_model.inference import InferenceModel_conditon_spec as score_old
from RiboDecode.check import Check


base_dir = os.path.dirname(__file__)
shutil.copytree(base_dir, os.path.abspath(os.path.curdir), dirs_exist_ok=True)
base_dir = os.path.abspath(os.path.curdir)
print(f"base dir: {base_dir}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def r2_score_func(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1 - b / (e + 1e-12)
    return f


dict_bp = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}

dict_id2bp = {
    0: "A",
    1: "C",
    2: "G",
    3: "T",
}


list_key2bp = []
for i in range(65):
    temp_key = dict_vocab_inv[i]
    temp_key2bp = []
    for temp_item_key in temp_key:
        temp_key2bp.append(dict_bp[temp_item_key])
    list_key2bp.append(temp_key2bp)

W_K2BP = torch.from_numpy(np.array(list_key2bp)).float().to(device)


with open(base_dir + "/data/pcscg.log", mode="r") as f:
    pcscg_info = f.readlines()[1:]
list_pcscg_info = []
list_codon_index = []
for line in pcscg_info:
    line = line.strip().split("\t")
    list_pcscg_info.append(line[1:])
    list_codon_index.append(line[0])
pcscg_info = np.array(list_pcscg_info).astype(np.float64)
max_value_cscg = np.max(pcscg_info)


list_pcscg_sub = []
for i in range(50):
    list_pcscg_sub.append(pcscg_info[:, i])


list_pcscg_W = []
for pcscg_sub in list_pcscg_sub:
    temp_pcscg_W = torch.zeros(size=(65, 1))
    for i, pcscg_v in enumerate(pcscg_sub):
        pcscg_v = float(pcscg_v)
        temp_pcscg_W[my_vocab[list_codon_index[i]]] = pcscg_v

    list_pcscg_W.append(temp_pcscg_W.to(device))


class seq_codon_gen(nn.Module):
    def __init__(self, model_config):
        super(seq_codon_gen, self).__init__()
        self.seqs = nn.Parameter(
            torch.randn(
                model_config["optim_batchsize"], model_config["max_len"] // 3, 65
            ),
            requires_grad=True,
        )

    def generate(self, codon_mask):
        self.seqs_codon = torch.exp(self.seqs) * codon_mask
        self.seqs_codon = self.seqs_codon / torch.sum(
            self.seqs_codon, dim=-1, keepdim=True
        )
        return self.seqs_codon.permute(0, 2, 1)

    def inital_gen(self, seq, model_config):
        temp_seq = np.array(seq * model_config["optim_batchsize"])
        temp_seq = torch.from_numpy(temp_seq).to(self.seqs.device)
        temp_seq = F.one_hot(temp_seq.long(), 65).float()
        self.seqs = nn.Parameter(temp_seq, requires_grad=True)


def gen_train(args):
    len_gene_ori = model_config["len_gene_ori"]
    rna_condition = model_config["RNA_condition"]
    sampler_type = model_config["sampler_type"]
    save_log_dir = (
        "./results_"
        + model_config["initial_seq"]
        + "/"
        + rna_condition
        + "_optim_mfe_"
        + sampler_type
        + "/"
        + str(idx_optim)
        + "/"
        + args.save_dir
    )
    os.makedirs(save_log_dir, exist_ok=True)

    if idx_optim == 0:
        if model_config["initial_seq"] == "natural":
            dataset_rna = Dataset_rna(
                model_config,
                "train",
                status="train",
                data_scale=model_config["num_seqs_train_gen"],
                seq=args.cds_seq,
            )
        elif model_config["initial_seq"] == "LinearDesign":
            dataset_rna = Dataset_rna(
                model_config,
                "ld",
                status="train",
                data_scale=model_config["num_seqs_train_gen"],
            )
    else:
        dataset_rna = Dataset_rna(
            model_config,
            "optim",
            status="train",
            data_scale=model_config["num_seqs_train_gen"],
            dist=data_gen_optim["mask_optim_dist"][idx_optim - 1],
        )
    loader_train = DataLoader(
        dataset_rna,
        batch_size=model_config["batch_size"],
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    list_seqs = []

    for _index, data in enumerate(loader_train):
        seq = data["seq_ori"].to(device)
        seq_con = data["seq_con"].to(device)
        seq_con_ = torch.argmax(seq_con.permute(0, 2, 1), dim=-1)
        list_seqs += seq_con_.detach().cpu().numpy().tolist()

    list_lines = []
    list_seqs_random = random.choices(list_seqs, k=50000)
    with open(save_log_dir + "/train_data_generate.txt", mode="w") as w:
        for temp_seq in list_seqs_random:
            seq_str = ""
            for id_seq in temp_seq:
                if round(id_seq, 0) in dict_vocab_inv.keys():
                    seq_str += dict_vocab_inv[round(id_seq, 0)]
                else:
                    seq_str += dict_vocab_inv[64]
            list_lines.append(seq_str[:len_gene_ori])
            w.write(seq_str[:len_gene_ori] + "\n")

    if idx_optim == 0:
        temp_list_data = list_lines[: model_config["num_seqs_select_mfe"]]
    else:
        sampler_type = model_config["sampler_type"]
        if sampler_type == "random":
            temp_list_data = list_lines[: model_config["num_seqs_select_mfe"] // 2]
        elif sampler_type == "optim":
            temp_list_data = data_gen_optim["seqs_optim_gen"][idx_optim - 1][
                : model_config["num_seqs_select_mfe"] // 2
            ]
        elif sampler_type == "random-optim":
            temp_list_data = (
                data_gen_optim["seqs_optim_gen"][idx_optim - 1][
                    : model_config["num_seqs_select_mfe"] // 4
                ]
                + list_lines[: model_config["num_seqs_select_mfe"] // 4]
            )
        elif sampler_type == "dist":
            temp_list_data = list_lines[: model_config["num_seqs_select_mfe"] // 2]
        elif sampler_type == "dist-optim":
            temp_list_data = (
                list_lines[: model_config["num_seqs_select_mfe"] // 4]
                + data_gen_optim["seqs_optim_gen"][idx_optim - 1][
                    : model_config["num_seqs_select_mfe"] // 4
                ]
            )
        else:
            temp_list_data = (
                data_gen_optim["seqs_optim_gen"][idx_optim - 1][
                    : model_config["num_seqs_select_mfe"] // 4
                ]
                + list_lines[: model_config["num_seqs_select_mfe"] // 4]
            )

    if model_config["mfe_tool_type"] == "rnafold":
        list_data_mfe = get_mfe(
            temp_list_data, len_sub=model_config["mfe_tool_sub_len"]
        )
    elif model_config["mfe_tool_type"] == "linearfold":
        list_data_mfe = get_mfe_lf(temp_list_data)

    list_data_gen = []
    with open(save_log_dir + "/train_data_generate_mfe.txt", mode="w") as w:
        for seq, mfe in list_data_mfe:
            temp_line = seq + "," + str(mfe) + "\n"
            list_data_gen.append([seq, mfe])
            w.write(temp_line)

    if idx_optim == 0:
        list_data_gen_sort = sorted(
            list_data_gen[:], key=lambda x: x[-1], reverse=False
        )
        data_gen_optim["data_gen_mfe"][idx_optim] = list_data_gen_sort[:]
        list_mfes = []
        for _, mfe in list_data_gen_sort:
            list_mfes.append(float(mfe))
    else:
        list_data_gen_sort = sorted(
            list_data_gen[:], key=lambda x: x[-1], reverse=False
        )
        if model_config["initial_seq"] == "LinearDesign":
            temp_list_data = (
                list_data_gen_sort[:][:-500] + data_gen_optim["data_gen_mfe"][0][:500]
            )
        else:
            temp_list_data = list_data_gen_sort[:]
        list_data_gen_sort = sorted(
            temp_list_data[:], key=lambda x: x[-1], reverse=False
        )
        data_gen_optim["data_gen_mfe"][idx_optim] = list_data_gen_sort

        list_mfes = []
        for _, mfe in list_data_gen_sort:
            list_mfes.append(float(mfe))


def get_sub_seqs(seq, len_sub=768):
    sub_seqs = []
    for i in range(0, len(seq), len_sub):
        sub_seqs.append(seq[i : i + len_sub])
    return sub_seqs


def get_eng_for_get_mfe(seq, len_sub):
    if len(seq) <= len_sub:
        _, temp_mfe = RNA.fold(seq)
    else:
        sub_seqs = get_sub_seqs(seq, len_sub)
        temp_mfe = 0
        for temp_seq in sub_seqs:
            _, t_f = RNA.fold(temp_seq)
            temp_mfe += t_f
    return [seq, temp_mfe]


def get_mfe(list_data, len_sub=4000):
    list_data_new = []
    for temp_data in list_data:
        if isinstance(temp_data, str):
            list_data_new.append(temp_data)
        else:
            list_data_new.append(temp_data[0])
    list_data = list_data_new
    list_data_mfe = []

    with Pool(cpu_count() - 1) as p:
        results = p.starmap(get_eng_for_get_mfe, [(seq, len_sub) for seq in list_data])

    for res in results:
        list_data_mfe.append(res)
    return list_data_mfe


def get_eng_for_get_mfe_lf(seq):
    os.makedirs("/tmp/LinearFold", exist_ok=True)
    r = os.popen("echo %s | /tmp/LinearFold/linearfold" % seq)
    info = r.readlines()[-1].split(" ")[-1][1:-2]
    mfe = float(info)
    return [seq, mfe]


def get_mfe_lf(list_data):
    list_data_new = []
    for temp_data in list_data:
        if isinstance(temp_data, str):
            list_data_new.append(temp_data)
        else:
            list_data_new.append(temp_data[0])
    list_data = list_data_new
    list_data_mfe = []

    with Pool(cpu_count() - 1) as p:
        results = p.map(get_eng_for_get_mfe_lf, list_data)

    for res in results:
        list_data_mfe.append(res)
    return list_data_mfe


def get_eng_for_get_mfe_sim(seq):
    _, temp_mfe = RNA.fold(seq)
    return temp_mfe


def get_mfe_sim(list_data):
    list_data_new = []
    for temp_data in list_data:
        if isinstance(temp_data, str):
            list_data_new.append(temp_data)
        else:
            list_data_new.append(temp_data[0])
    list_data_mfe = []

    with Pool(cpu_count() - 1) as p:
        results = p.map(get_eng_for_get_mfe_sim, list_data_new)

    for res in results:
        list_data_mfe.append(res)
    return list_data_mfe


def mfe_train(args):
    print("optim mfe-prediction model ... ...")
    rna_condition = model_config["RNA_condition"]
    sampler_type = model_config["sampler_type"]
    save_log_dir = (
        "./results_"
        + model_config["initial_seq"]
        + "/"
        + rna_condition
        + "_optim_mfe_"
        + sampler_type
        + "/"
        + str(idx_optim)
        + "/"
        + args.save_dir
    )
    os.makedirs(save_log_dir, exist_ok=True)

    list_data_mfe = []
    if idx_optim == 0:
        list_data_mfe = data_gen_optim["data_gen_mfe"][idx_optim]
    else:
        list_data_mfe = data_gen_optim["data_gen_mfe"][idx_optim]

    print("train seqs num for MFE:", len(list_data_mfe) / 10000, "w")

    if idx_optim == 0:
        dataset_mfe = Dataset_rna_mfe(
            model_config,
            list_data_mfe,
            data_scale=model_config["num_seqs_train_mfe"] * 2,
        )
    else:
        dataset_mfe = Dataset_rna_mfe(
            model_config,
            list_data_mfe,
            data_scale=model_config["num_seqs_train_mfe"] // 2,
        )
    loader_train = DataLoader(
        dataset_mfe,
        batch_size=model_config["batch_size"],
        shuffle=True,
        num_workers=model_config["num_workers"],
        pin_memory=True,
    )

    learning_rate = model_config["lr"]

    if idx_optim == 0:
        optim = torch.optim.AdamW(
            model_mfe.parameters(), learning_rate, weight_decay=5e-4
        )
    else:
        optim = torch.optim.AdamW(
            model_mfe.parameters(), learning_rate * 0.1, weight_decay=5e-4
        )
    loss_fun_l1 = nn.SmoothL1Loss()

    model_mfe.train()
    list_mfe_preds = []
    list_mfe_targets = []

    fgm = FGM(model_mfe)

    for _index, data in enumerate(loader_train):
        seq = data["seq"].to(device)
        mfe = data["mfe"].to(device)

        optim.zero_grad()
        out_mfe = model_mfe(seq)
        loss_mfe = loss_fun_l1(out_mfe.view(-1), mfe)

        temp_a = float(random.randint(100, 1000) / 1000.0)
        fgm.attack(epsilon=temp_a, emb_name="encoder")
        pred_adv = model_mfe(seq)
        loss_adv = loss_fun_l1(pred_adv.view(-1), mfe)
        fgm.restore(emb_name="encoder")

        loss = loss_mfe + loss_adv
        loss.backward()
        optim.step()
        list_mfe_preds += out_mfe.view(-1).detach().cpu().numpy().tolist()
        list_mfe_targets += mfe.view(-1).detach().cpu().numpy().tolist()

    model_mfe.eval()
    torch.save(model_mfe.state_dict(), save_log_dir + "epoch_temp_mfe.pb")
    print(
        "R2 of train data for MFE:{:.2f}".format(
            r2_score_func(np.array(list_mfe_targets[:]), np.array(list_mfe_preds[:]))
        )
    )


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                with suppress(Exception):
                    norm = torch.norm(param.grad)
                    if norm != 0 and not torch.isnan(norm):
                        r_at = epsilon * param.grad / norm
                        param.data.add_(r_at)

    def restore(self, emb_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def optim(args):
    batch_size = model_config["optim_batchsize"]
    # data_dir = model_config["data_folder"] + "/train_data.txt"
    len_gene_ori = model_config["len_gene_ori"]
    print("length of original sequence:", len_gene_ori)
    res_k, _ = read_lines([args.cds_seq.replace("U", "T")], model_config["max_len"])

    list_res_ks = []
    for temp_k in res_k[0]:
        codon_ks = dict_codon_group[temp_k]
        temp_res_k = np.array([0] * 65)
        for codon_k in codon_ks:
            temp_res_k[codon_k] = 1
        list_res_ks.append(temp_res_k)
    mask_gene_codon = np.array([list_res_ks])
    mask_gene_codon = torch.from_numpy(mask_gene_codon).to(device).float()

    if idx_optim == 0 and model_config["initial_seq"] == "LinearDesign":
        data_dir = model_config["data_folder"] + "/ld_data.txt"
        res_k, _ = read_data(data_dir, model_config["max_len"])
    else:
        pass

    rna_condition = model_config["RNA_condition"]
    rna_condition_neg = model_config["RNA_condition_negative"]
    sampler_type = model_config["sampler_type"]
    result_fold = (
        "./results_"
        + model_config["initial_seq"]
        + "/"
        + rna_condition
        + "_optim_mfe_"
        + sampler_type
        + "/"
        + str(idx_optim)
        + "/"
    )
    result_dir = (
        "./results_"
        + model_config["initial_seq"]
        + "/"
        + rna_condition
        + "_optim_mfe_"
        + sampler_type
        + "/"
        + str(idx_optim)
        + "/"
        + args.result_dir
    )

    data_gen_optim["optim_iteration_logs"][idx_optim] = result_fold
    os.makedirs(result_dir, exist_ok=True)
    score_model = score_old(args.model_config_s, args.best_model)
    if model_config["using_custom_env"]:
        score_model.prepare(
            batch_size, rna_condition, base_dir + "/score_model/conditions/", args.csv
        )
    else:
        score_model.prepare(
            batch_size, rna_condition, base_dir + "/score_model/conditions/"
        )

    if idx_optim == 0:
        dataset_rna = Dataset_rna(
            model_config,
            dataset_name="train",
            status="train",
            data_scale=model_config["optim_num"],
            seq=args.cds_seq,
        )
    else:
        dataset_rna = Dataset_rna(
            model_config,
            dataset_name="optim",
            status="train",
            data_scale=model_config["optim_num"],
        )
    loader_valid = DataLoader(
        dataset_rna,
        batch_size=batch_size,
        shuffle=False,
        num_workers=model_config["num_workers"] // 8,
        drop_last=False,
    )

    optim = torch.optim.AdamW(
        z_model.parameters(), lr=model_config["optim_lr"], weight_decay=1e-4
    )
    seq_gen_pad = torch.zeros(batch_size, 4500 - model_config["max_len"], 4).to(device)
    temp_2 = len(loader_valid) // 2
    temp_4 = temp_2 // 2
    temp_8 = temp_4 // 2
    temp_16 = temp_8 // 2

    lr_sch = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[temp_2, temp_4 * 3, temp_8 * 7, temp_16 * 17]
    )

    list_index = []
    list_loss = []
    list_rpf = []
    list_seqs = []
    list_similarity = []
    list_mfe = []
    list_seq_gens = []
    list_cscgs = []

    order_windows_list = []
    for i in range(0, 99, 2):
        order_windows_list.append([i / 100, (i + 2) / 100])

    model_mfe.eval()
    for i, data in enumerate(loader_valid):
        seq = data["seq_ori"].to(device)
        b = seq.shape[0]
        seq_gen = z_model.generate(mask_gene_codon.clone())
        list_seq_gens.append(seq_gen.detach().cpu())

        temp_seq_gen = F.one_hot(torch.argmax(seq_gen, 1), 65).float()
        temp_seq_gen = temp_seq_gen.permute(0, 2, 1) - seq_gen.detach() + seq_gen
        temp_score_mfe = model_mfe(temp_seq_gen)

        loss_mfe = torch.mean(
            -float(model_config["mfe_norm_index"]) / (temp_score_mfe)
        )
        score_mfe = temp_score_mfe

        list_temp_cscgs = []
        l = model_config["len_gene_ori"] // 3
        for n, idx in enumerate(order_windows_list):
            start = int(idx[0] * l)
            end = int(idx[1] * l)
            list_temp_cscgs.append(
                torch.matmul(
                    temp_seq_gen.permute(0, 2, 1)[:, start:end, :], list_pcscg_W[n]
                ).view(b, -1)
            )

        temp_cscgs = torch.sum(torch.concatenate(list_temp_cscgs, dim=1), dim=1)
        # loss_cscgs = max_value_cscg * l - torch.mean(temp_cscgs)

        mean_sim = torch.mean(torch.cosine_similarity(seq, seq_gen), dim=1)
        # loss_sim = torch.mean(1 - torch.sqrt(torch.pow(mean_sim, 2)))

        seq_gen_label = torch.argmax(seq_gen, 1)
        seq_gen_onehot = F.one_hot(seq_gen_label, 65).float()
        seq_gen = seq_gen.permute(0, 2, 1)
        seq_gen_new = seq_gen_onehot + seq_gen - seq_gen.detach()

        temp_codon = torch.sum(seq_gen_new * mask_gene_codon) / (
            batch_size * model_config["max_len"] // 3
        )
        loss_codon = 1 - temp_codon

        seq_gen_bp = torch.matmul(seq_gen_new, W_K2BP.view(-1, 12)).view(b, -1, 4)[
            :, :4500, :
        ]
        seq_gen_bp_pad = torch.cat([seq_gen_bp, seq_gen_pad], dim=1)

        score_target, score_neg = score_model.predict_seq_spec_single(
            seq_gen_bp_pad.permute(0, 2, 1), rna_condition, rna_condition_neg
        )
        rpf_target = 0.2 * torch.expm1(score_target)
        loss_target = torch.mean(
            torch.sqrt(torch.pow(model_config["rpf_target"] - rpf_target, 2)) / 10.0
        )
        # rpf_negative = 0.2 * torch.expm1(score_neg)
        # loss_rpf = loss_target + loss_neg * 0.7

        if model_config["rpf_target"] == 100:
            loss_rpf = loss_target * 0.1
        else:
            loss_rpf = loss_target

        loss = loss_codon + loss_mfe * weight_mfe + loss_rpf * (1 - weight_mfe)
        loss.backward()

        optim.step()
        optim.zero_grad()

        lr_sch.step()

        list_loss += [loss.detach().cpu().item()] * batch_size
        list_rpf += (
            (torch.expm1(score_target.detach().cpu()) / 5).view(-1).numpy().tolist()
        )
        list_similarity += mean_sim.detach().cpu().tolist()
        list_mfe += score_mfe.detach().cpu().view(-1).numpy().tolist()
        out_rec = torch.argmax(seq_gen, dim=-1)
        list_seqs += out_rec.detach().cpu().numpy().tolist()
        list_cscgs += temp_cscgs.detach().cpu().numpy().tolist()

    for i in range(len(list_loss)):
        list_index.append(i)

    list_seqs_str = []
    list_rpf_true = []
    list_index_t = []
    list_mfe_true = []
    list_seq_rpf_mfe = []
    with open(result_dir + "/optim_results.txt", mode="w") as w:
        id_line = 0
        for temp_seq in list_seqs[: model_config["optim_num"]]:
            seq_str = ""
            for id_seq in temp_seq:
                if round(id_seq, 0) in dict_vocab_inv.keys():
                    seq_str += dict_vocab_inv[round(id_seq, 0)]
                else:
                    seq_str += dict_vocab_inv[64]

            rpf_score = str(list_rpf[id_line])
            mfe_score = str(list_mfe[id_line])
            id_line += 1

            seq_str = seq_str[:len_gene_ori]
            error_num = 0
            if error_num == 0:
                list_rpf_true.append(float(rpf_score))
                list_mfe_true.append(float(mfe_score))
                list_index_t.append(id_line)
                list_seq_rpf_mfe.append([seq_str, float(rpf_score), float(mfe_score)])
                w.write(seq_str + "\t" + rpf_score + "\t" + mfe_score + "\n")
                list_seqs_str.append(seq_str)
            else:
                pass

    list_rpf_base = [list_rpf[0]] * len(list_rpf)
    data_gen_optim["data_optim_rpf"][idx_optim] = list_rpf
    if idx_optim == 0:
        data_gen_optim["data_rpf_base"] = list_rpf_base

    list_mfe_base = [list_mfe[0]]*len(list_mfe)
    data_gen_optim['rna_condition'] = rna_condition
    data_gen_optim['data_optim_mfe'][idx_optim] = list_mfe
    if idx_optim == 0:
        data_gen_optim['data_mfe_base'] = list_mfe_base

    data_gen_optim["data_optim_cscg"][idx_optim] = list_cscgs
    print("generate different seqs num:", len(set(list_seqs_str)))
    check.calc_mean_error_rate(result_dir + "/optim_results.txt")

    temp_num_seqs = len(list_seq_rpf_mfe)

    list_seq_mfe_sort = sorted(
        list_seq_rpf_mfe[1 * temp_num_seqs // 4 :], key=lambda x: x[-1], reverse=False
    )[:]

    data_gen_optim["seqs_optim_gen"][idx_optim] = list_seq_mfe_sort
    temp_best_mfe = 0
    temp_best_idx = 0
    temp_best_rpf = 0
    temp_list_seqs = list_seq_mfe_sort[: model_config["num_optim_top"]]
    temp_list_mfes = get_mfe_sim(temp_list_seqs)

    for i in range(len(temp_list_mfes)):
        temp_mfe = temp_list_mfes[i]
        temp_rpf = temp_list_seqs[i][1]
        if weight_mfe == 1:
            if temp_mfe <= temp_best_rpf:
                temp_best_mfe = temp_mfe
                temp_best_idx = i
                temp_best_rpf = temp_rpf
        else:
            if temp_rpf >= temp_best_rpf:
                temp_best_mfe = temp_mfe
                temp_best_idx = i
                temp_best_rpf = temp_rpf

    temp_seq_optim_new = list_seq_mfe_sort[temp_best_idx][0]
    temp_seq_optim_mfe_model = float(list_seq_mfe_sort[temp_best_idx][-1])
    temp_seq_optim_mfe_tool = temp_best_mfe
    temp_seq_optim_rpf_model = float(list_seq_mfe_sort[temp_best_idx][1])
    print("seq mfe model:", temp_seq_optim_mfe_model)
    print("seq mfe tool :", temp_seq_optim_mfe_tool)
    print("seq rpf model:", temp_seq_optim_rpf_model)

    data_gen_optim["results_rpf"][idx_optim] = temp_seq_optim_rpf_model
    data_gen_optim["results_mfe"][idx_optim] = [
        temp_seq_optim_mfe_model,
        temp_seq_optim_mfe_tool,
    ]
    data_gen_optim["seq_optim_best"][idx_optim] = temp_seq_optim_new

    data_dir = model_config["data_folder"] + "/optim_data.txt"
    with open(data_dir, mode="w") as w:
        w.write(temp_seq_optim_new + "\n")

    temp_optim_seqs = torch.concatenate(list_seq_gens[-5:], dim=0).permute(0,2,1).detach().cpu().numpy()
    data_gen_optim['mask_optim_dist'][idx_optim] = np.mean(temp_optim_seqs, axis=0).tolist()


parser = argparse.ArgumentParser()

parser.add_argument("--cds", type=str)
parser.add_argument("--cds_seq", type=str, required=True)
parser.add_argument("--alpha", type=float)
parser.add_argument("--beta", type=float)
parser.add_argument("--mfe_weight", type=float, default=0.7, help="float in [0, 1]")
parser.add_argument("--env", type=str)
parser.add_argument("--optim_epoch", type=int, default=20, help="optim epoch num")
parser.add_argument("--csv", type=str)

args = parser.parse_args()
print(args)

os.makedirs(os.path.join(base_dir, "data", "cds"), exist_ok=True)
# for n in ["train_data.txt", "ld_data.txt"]:
#     with open(os.path.join(base_dir, "data", "cds", n), 'w') as f:
#         f.write(args.cds)

args.model_config_g = base_dir + "/model_config.py"
args.save_dir = "./logs/"
args.result_dir = "./samples/"
args.model_config_s = base_dir + "/score_model/model_config.json"
args.best_model = base_dir + "/score_model/best_model.p"

data_gen_optim = {}
data_gen_optim["data_gen_mfe"] = {}
data_gen_optim["data_optim_rpf"] = {}
data_gen_optim["data_optim_mfe"] = {}
data_gen_optim["data_optim_cscg"] = {}
data_gen_optim["results_rpf"] = {}
data_gen_optim["results_mfe"] = {}
data_gen_optim["results_cscg"] = {}
data_gen_optim["seqs_optim_gen"] = {}
data_gen_optim["mask_optim_dist"] = {}
data_gen_optim["seq_optim_best"] = {}
data_gen_optim["optim_iteration_logs"] = {}

model_config = json.loads(open(base_dir + "/model_config.py", mode="r").read())[
    "training"
]
model_config["data_folder"] = base_dir + "/data/cds/"
model_config["weight_mfe"] = args.mfe_weight
model_config["num_optim"] = args.optim_epoch
model_config["RNA_condition"] = args.env
model_config["rpf_target"] = args.alpha
model_config["mfe_norm_index"] = args.beta
model_config["using_custom_env"] = False
if args.env not in ['HEK293T', 'BJ', 'A549', 'HeLa']:
    model_config["using_custom_env"] = True
orig_seq = args.cds_seq.replace("U", "T")
len_gene_ori = len(orig_seq)

check = Check(orig_seq)

model_config["len_gene_ori"] = len_gene_ori
model_config["max_len"] = len_gene_ori
print("max_len:", model_config["max_len"])
model_mfe = mfe_conv_sim(65, model_config["hidden_dim"], model_config["latent_dim"]).to(
    device
)
z_model = seq_codon_gen(model_config).to(device)

weight_mfe = model_config["weight_mfe"]
num_optim = model_config["num_optim"]
for idx_optim in range(num_optim):
    time_start = time.time()
    if model_config["weight_mfe"] > 0:
        gen_train(args)
        mfe_train(args)
    optim(args)

    print("loop time cost:{:.0f}s".format(time.time() - time_start))
    print(
        "=================================================================================================================================loop: {} finished!".format(
            idx_optim + 1
        )
    )

for i in range(num_optim):
    print(i + 1, data_gen_optim["results_rpf"][i])
for i in range(num_optim):
    print(i + 1, data_gen_optim["results_mfe"][i])
