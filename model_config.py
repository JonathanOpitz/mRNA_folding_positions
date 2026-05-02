{
    "training": {
        "data_folder": "./data/gluc_cds/",
        "epochs": 1,
        "batch_size": 128,
        "num_workers": 8,
        "codon_style": "all",
        "lr": 1e-4,
        "max_len": 768,

        "num_layers": 4,
        "dim_embedding": 256,
        "n_heads": 4,
        "dim_forward": 512,
        "dropout": 0.1,
        "n_token": 65,

        "latent_dim": 256,
        "hidden_dim": 512,

        "inference_num": 1000,
        "RNA_condition": "HEK293T",
        "RNA_condition_help": ["HEK293T, A549, BJ"],
        "RNA_condition_negative": "A549",
        "optim_num": 40000,

        "initial_seq": "natural",
        "initial_seq_help": ["natural", "LinearDesign", "other"],

        "optim_lr": 0.1,
        "optim_batchsize": 20,
        "optim_target": "mfe",
        "num_seqs_train_gen": 100000,
        "num_seqs_select_mfe": 20000,

        "mfe_tool_type": "rnafold",
        "mfe_tool_type_help": ["rnafold", "linearfold", "other"],
        "mfe_tool_sub_len": 768,

        "num_seqs_train_mfe": 500000,
        "num_optim": 20,
        "num_optim_top": 480,

        "rpf_target": 100,
        "weight_mfe": 0.7,
        "mfe_norm_index": 100,
        "mfe_norm_index_help": [10, 100, 1000],
        "weight_model_dir": "",

        "sampler_type": "dist-optim",
        "sampler_help": ["random", "optim", "random-optim", "dist", "dist-optim"],
        "": ""

    }
}
