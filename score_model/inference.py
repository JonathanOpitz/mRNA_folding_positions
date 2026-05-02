import torch
import torch.nn as nn
import json
import argparse
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import os
import csv
import pandas as pd


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 2, 1)


class CNN_Encoder_Model(nn.Module):
    def __init__(self, model_config, input_shapes):
        super(CNN_Encoder_Model, self).__init__()
        self.model_config = model_config
        self.input_shapes = input_shapes
        hyper_use = model_config['training']['hyper_use']
        p = model_config[hyper_use]
        self.motif_depth = 4
        self.bias = nn.Parameter(torch.randn([1]))

        self.cds_detector_list = nn.ModuleList()
        for i in range(self.motif_depth):
            self.cds_detector_list.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=input_shapes[0][0],
                              out_channels=int(p['filters1']),
                              kernel_size=int(p['kernel_size1']),
                              stride=int(p['filters_stride1']),
                              dilation=int(p["dilation1"]),
                              ),
                    nn.ReLU(),
                    nn.BatchNorm1d(int(p['filters1'])),
                )
            )

        self.cds_filter = nn.Sequential(
            nn.BatchNorm1d(int(p['filters1'])),
            nn.Dropout1d(float(p['dropout1'])),
            nn.MaxPool1d(kernel_size=int(p['pool_size1']), stride=int(p['stride1']), padding=0)
        )

        self.cds_encoder = nn.Sequential(
            nn.Conv1d(in_channels=int(p['filters1']),
                      out_channels=int(p['filters3']),
                      kernel_size=int(p['kernel_size3']),
                      stride=int(p['filters_stride3']),
                      dilation=int(p["dilation3"]),
                      ),
            nn.ReLU(),
            nn.BatchNorm1d(int(p['filters3'])),
            nn.MaxPool1d(kernel_size=int(p['pool_size3']), stride=int(p['stride3']), padding=0),
            Transpose(),
            nn.Dropout1d(float(p['dropout3'])),
            nn.Flatten()
        )

        cds_flatten_size = self.cds_conv_shape()

        self.RPF_fc = nn.Sequential(
            nn.Linear(in_features=cds_flatten_size, out_features=int(p['dense5'])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p['dense5'])),
            nn.Dropout(float(p['dropout5'])),

            nn.Linear(in_features=int(p['dense5']), out_features=1)
        )

        self.attention_cds = nn.Sequential(
            nn.Linear(in_features=int(p['dense6']) + 1, out_features=int(p['filters1']) * self.motif_depth),
            nn.Tanh()
        )

        self.mRNA_layer = nn.Sequential(
            nn.Linear(in_features=self.input_shapes[2][0], out_features=int(p['dense4'])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p['dense4'])),
            nn.Dropout(float(p['dropout4'])),
            nn.Linear(in_features=int(p['dense4']), out_features=int(p['dense6'])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p['dense6'])),
        )

    def cds_motif_detection(self, sequence_input, a):
        results = [motif_detector(sequence_input) for motif_detector in self.cds_detector_list]
        avg_filter = torch.exp(a)
        features = torch.sum(torch.stack(results, 3) * avg_filter, 3) / torch.sum(avg_filter, 3)
        motif_result = self.cds_filter(features)

        return motif_result

    def cds_conv_shape(self):
        x_input_1 = torch.zeros(1, *self.input_shapes[1])
        x_output = self.cds_motif_detection(x_input_1, torch.zeros([1, 256, 1, self.motif_depth]))
        x_output = self.cds_encoder(x_output)
        flatten_shape = int(np.prod(x_output.size()))

        return flatten_shape

    def forward(self, cds_sequence, mRNA_array, mRNA_count):
        mRNA_features = self.mRNA_layer(mRNA_array)
        attention_cds = torch.reshape(self.attention_cds(torch.concat([mRNA_features, mRNA_count], 1)), [-1, 256, 1, self.motif_depth])
        cds_output = self.cds_motif_detection(cds_sequence, attention_cds)
        cds_seq_features = self.cds_encoder(cds_output)
        RPF = self.RPF_fc(cds_seq_features) * mRNA_count + self.bias

        return RPF


class InferenceModel:

    input_shape = [[4, 5000], [4, 4500], [10552, ], [1, ], [1, ]]

    def __init__(self, model_config: str, best_model: str):
        self.model_config = json.load(open(model_config, 'r'))
        self.predictorModel = CNN_Encoder_Model(model_config=self.model_config, input_shapes=InferenceModel.input_shape)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__load_best_model_weights__(best_model=best_model)
        self.predictorModel.to(self.device)
    
    def __load_best_model_weights__(self, best_model):
        checkpoint = torch.load(best_model, map_location='cpu')
        self.predictorModel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Successfully loaded weights from {}".format(best_model))

    def load_in_cds(self, f_name):
        max_cds_len = 4500
        with open(f_name, "r") as f:
            cds_lines = f.readlines()
        cds_lines = [i.strip("\n").split("\t")[0].upper() for i in cds_lines]
        # cds_lines = [i.strip().upper() for i in cds_lines]
        raw_lines = []
        for index, cds in enumerate(cds_lines):
            len_cds = len(cds)
            if len_cds < max_cds_len:
                need_padding_num = max_cds_len - len_cds
                cds = cds + "N" * need_padding_num
            elif len_cds > max_cds_len:
                cds = cds[:max_cds_len]
            raw_lines.append(cds)

        return raw_lines

    def my_process_line_for_cds(self, line: str, max_len: int = None) -> np.ndarray:
        dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        dna_I = np.eye(len(dna_vocab))
        try:
            res = []
            for i in line:
                res.append(dna_I[dna_vocab[i]])
            if max_len:
                if len(res) < max_len:
                    need_padding_num = max_len - len(res)
                    res = res + [dna_I[dna_vocab["N"]]] * need_padding_num
                else:
                    res = res[:max_len]
            arr = np.array(res)
        except Exception as exc:
            print(exc)
            raise Exception("Unable to process line: {}".format(line))

        return np.expand_dims(arr, 0)
    
    def read_csv_data(self, file_path):
        max_cds_len = 4500
        cds_data = []
        log_rpf_data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                cds_data.append(row[0])
                log_rpf_data.append(row[1])
        raw_lines = []
        for index, cds in enumerate(cds_data):
            len_cds = len(cds)
            if len_cds < max_cds_len:
                need_padding_num = max_cds_len - len_cds
                cds = cds + "N" * need_padding_num
            elif len_cds > max_cds_len:
                cds = cds[:max_cds_len]
            raw_lines.append(cds)
        return raw_lines, log_rpf_data

    def read_npz_data(self, file_path):
        max_cds_len = 4500
        cds_file = np.load(file_path, allow_pickle=True)
        cds_data = cds_file['cds_seq'].tolist()
        raw_lines = []
        for index, cds in enumerate(cds_data):
            len_cds = len(cds)
            if len_cds < max_cds_len:
                need_padding_num = max_cds_len - len_cds
                cds = cds + "N" * need_padding_num
            elif len_cds > max_cds_len:
                cds = cds[:max_cds_len]
            raw_lines.append(cds)
        return raw_lines
    
    def read_csv_internal_data(self, file_path):
        max_cds_len = 4500
        cds_data = []
        rpf_data = []
        RNA_counts_data = []
        mRNA_data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                RNA_counts_data.append(row[0])
                rpf_data.append(row[1])
                cds_data.append(row[2])
                mRNA_data.append(row[3:])
        RNA_counts_data = np.array(RNA_counts_data, dtype=float)
        RNA_counts_data = np.log1p(RNA_counts_data * 5).reshape(-1, 1)
        rpf_data = np.array(rpf_data, dtype=float)
        rpf_data = np.log1p(rpf_data * 5)
        mRNA_data = [np.array(row, dtype=float) for row in mRNA_data]
        mRNA_data = [np.log1p(row * 5) for row in mRNA_data]
        cds_data_pad = []
        for index, cds in enumerate(cds_data):
            len_cds = len(cds)
            if len_cds < max_cds_len:
                need_padding_num = max_cds_len - len_cds
                cds = cds + "N" * need_padding_num
            elif len_cds > max_cds_len:
                cds = cds[:max_cds_len]
            cds_data_pad.append(cds)
        return RNA_counts_data, rpf_data, cds_data_pad, mRNA_data

    def __create_model_input_ex__(self, npzFilename):
        npzfile = np.load(npzFilename, allow_pickle=True)
        inputNameList = ['mRNA']
        modelInput = OrderedDict({k: deepcopy(npzfile[k].astype(np.float32)) for k in inputNameList})
        for k in ['mRNA']:
            modelInput[k] = np.expand_dims(modelInput[k], 0)
            modelInput[k] = np.log1p(modelInput[k] * 5)
        modelInput['mRNA'] = modelInput['mRNA'][:, :10552]
        
        return modelInput
    
    def predict_internal_data(self, Filename, zero_out=('RNA_counts', 'cds', 'mRNA')):
        batch_size = 256
        RNA_counts_data, rpf_data, cds_data_pad, mRNA_data = self.read_csv_internal_data(Filename)
        self.predictorModel.eval()
        scores = []
        for i in range(0, len(cds_data_pad), batch_size):
            RNA_counts = RNA_counts_data[i: i + batch_size]
            cds = cds_data_pad[i: i + batch_size]
            mRNA = mRNA_data[i: i + batch_size]
            cds_oh = [self.my_process_line_for_cds(s) for s in cds]

            if 'RNA_counts' in zero_out:
                RNA_counts = np.full_like(RNA_counts, 4.5)
            if 'cds' in zero_out:
                cds_oh = np.zeros_like(cds_oh)
            if 'mRNA' in zero_out:
                mRNA = np.zeros_like(mRNA)

            with torch.no_grad():
                input_cds = torch.Tensor(np.vstack(cds_oh))
                cds_sequence = torch.transpose(input_cds, 2, 1)[:, :4, :].to(self.device)
                mRNA_array = torch.Tensor(mRNA).to(self.device)
                mRNA_count = torch.Tensor(RNA_counts).to(self.device)
                prediction = self.predictorModel(cds_sequence, mRNA_array, mRNA_count)
                scores = scores + prediction.data.cpu().numpy()[:, 0].tolist()

        return scores, rpf_data
    
    def predict(self, InputFilename, Filename):
        batch_size = 256
        modelInput = self.__create_model_input_ex__(InputFilename)
        mRNA_array = [v for v in modelInput.values()]
        mRNA_array = np.squeeze(mRNA_array, axis=1)
        mRNA_count = np.array([[4.5]])
        cds_str = self.load_in_cds(Filename)
        # cds_str, log_rpf = self.read_csv_data(Filename)
        # cds_str = self.read_npz_data(Filename)
        self.predictorModel.eval()
        scores = []
        for i in range(0, len(cds_str), batch_size):
            cds = cds_str[i: i + batch_size]
            cds_num = len(cds)
            cds_oh = [self.my_process_line_for_cds(s) for s in cds]
            with torch.no_grad():
                input_cds = torch.Tensor(np.vstack(cds_oh))
                cds_sequence = torch.transpose(input_cds, 2, 1)[:, :4, :].to(self.device)
                mRNA_array = torch.Tensor(mRNA_array).to(self.device)
                mRNA_array_batch = torch.cat([mRNA_array] * cds_num, dim=0)
                mRNA_count = torch.Tensor(mRNA_count).to(self.device)
                mRNA_count_batch = torch.cat([mRNA_count] * cds_num, dim=0)
                prediction = self.predictorModel(cds_sequence, mRNA_array_batch, mRNA_count_batch)
                scores = scores + prediction.data.cpu().numpy()[:, 0].tolist()

        return scores


class InferenceModel_conditon_spec:
    input_shape = [[4, 5000], [4, 4500], [10552, ], [1, ], [1, ]]

    def __init__(self, model_config: str, best_model: str):
        self.model_config = json.load(open(model_config, 'r'))
        self.predictorModel = CNN_Encoder_Model(model_config=self.model_config, input_shapes=InferenceModel.input_shape)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.__load_best_model_weights__(best_model=best_model)
        self.predictorModel.to(self.device)

    def __load_best_model_weights__(self, best_model):
        checkpoint = torch.load(best_model, map_location='cpu')
        self.predictorModel.load_state_dict(checkpoint['model_state_dict'], strict=False)

    def __create_model_input__(self, npzFilename):
        npzfile = np.load(npzFilename, allow_pickle=True)
        inputNameList = ['mRNA']
        modelInput = OrderedDict({k: deepcopy(npzfile[k].astype(np.float32)) for k in inputNameList})
        for k in ['mRNA']:
            modelInput[k] = np.expand_dims(modelInput[k], 0)
            modelInput[k] = np.log1p(modelInput[k] * 5)
        modelInput['mRNA'] = modelInput['mRNA'][:, :10552]

        return modelInput

    def prepare(self, batch_size=1, condition='retina', conditons_dir='./score_model/conditions/', custom_csv=None):

        list_conditions = os.listdir(conditons_dir)

        self.dict_rna_condition = {}
        self.conditions = []
        self.conditions_negative = []
        for temp_condition_npy in list_conditions:
                temp_condition = temp_condition_npy.split('.')[0].split('_')[-1]
                temp_modelInput = self.__create_model_input__(conditons_dir + temp_condition_npy)
                with torch.no_grad():
                    temp_mRNA_array = torch.Tensor(temp_modelInput['mRNA']).to(self.device)
                    temp_mRNA_array_batch = torch.cat([temp_mRNA_array] * batch_size, dim=0)
                self.dict_rna_condition[temp_condition] = temp_mRNA_array_batch

        with torch.no_grad():
            mRNA_count = np.array([[4.5]])
            self.mRNA_count = torch.Tensor(mRNA_count).to(self.device)
            self.mRNA_count_batch = torch.cat([self.mRNA_count] * batch_size, dim=0)

        for k in self.dict_rna_condition.keys():
            self.conditions.append(k)
            if k != condition:
                self.conditions_negative.append(k)

        if custom_csv:
            df_mRNA = pd.read_csv(custom_csv, index_col=0)
            df_mRNA.columns = ['rpkm']
            mRNA_counts = np.array(df_mRNA['rpkm']).astype(np.float32)
            mRNA_counts = np.expand_dims(mRNA_counts, 0)
            mRNA_counts = np.log1p(mRNA_counts * 5)
            mRNA_counts = mRNA_counts[:, :10552]
            with torch.no_grad():
                temp_mRNA_array = torch.Tensor(mRNA_counts).to(self.device)
                temp_mRNA_array_batch = torch.cat([temp_mRNA_array] * batch_size, dim=0)
            self.dict_rna_condition[condition] = temp_mRNA_array_batch
            self.conditions.append(condition)

    def predict_seq(self, cds_seq, condition='retina'):
        self.predictorModel.eval()
        prediction = self.predictorModel(cds_seq, self.dict_rna_condition[condition], self.mRNA_count_batch)
        return prediction

    def predict_seq_spec_single(self, cds_seq, condition='HEK293T', condition_neg='A549'):
        self.predictorModel.eval()
        prediction_target = self.predictorModel(cds_seq, self.dict_rna_condition[condition], self.mRNA_count_batch)
        prediction_negative = self.predictorModel(cds_seq, self.dict_rna_condition[condition_neg], self.mRNA_count_batch)

        return prediction_target, prediction_negative

    def predict_seq_spec_multi(self, cds_seq, condition='HEK293T'):
        self.predictorModel.eval()
        prediction_target = self.predictorModel(cds_seq, self.dict_rna_condition[condition], self.mRNA_count_batch)

        predictions_negative = []
        for temp_conditon in self.conditions_negative:
            predictions_negative.append(self.predictorModel(cds_seq, self.dict_rna_condition[temp_conditon], self.mRNA_count_batch))

        return prediction_target, predictions_negative

    def zero_grad(self):
        self.predictorModel.zero_grad()
        return True


def main(args):
    InferenceObject = InferenceModel(args.model_config, args.best_model)
    prediction = InferenceObject.predict(args.mRNA_file, args.cds_file)
    print(prediction)


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default=base_dir + '/model_config.json')
    parser.add_argument('--best_model', type=str, default=base_dir + '/best_model.p')
    parser.add_argument('--mRNA_file', type=str, default=base_dir + "/conditions/HEK293T_normal_10552_RPKM_HEK293T.npz")
    parser.add_argument("--cds_file", type=str, default=base_dir + "/test_cds.txt")
    args = parser.parse_args()
    main(args)
