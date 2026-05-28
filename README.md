# Predicting Translational Pause Sites in mRNA Using Graph Neural Networks

A machine-learning pipeline that predicts where ribosomes should pause during
translation, and uses those predictions to guide codon optimisation toward
sequences that support correct co-translational protein folding.

## Biological Problem

During translation, ribosomes do not synthesize proteins at a uniform rate. They
pause at specific codons, and growing evidence suggests these pauses are
functionally important: they give nascent protein domains time to fold
co-translationally before they emerge from the ribosome tunnel. This
co-translational folding is especially critical for proteins with complex
multi-domain architectures, where premature folding of one domain in the presence
of another can lead to aggregation or misfolding.

Beyond basic biology, this has direct relevance to mRNA vaccine and therapeutic
protein design — the codon usage of a synthetic mRNA determines ribosome
dynamics, which in turn affects protein yield and correct folding. Existing codon
optimisers, including RiboDecode, maximise ribosome load and mRNA stability but do
not account for co-translational folding. This project closes that gap: it
predicts *where* ribosomes should pause from protein structure and mRNA sequence,
and integrates that prediction into the optimiser as an additional objective.

## Project Overview

The project runs in four stages:

1. **Preprocessing** — build a per-codon feature set for each gene from Ribo-seq,
   AlphaFold2 structures, and RNA secondary structure, then combine them into a
   folding demand score.
2. **Folding demand model** — train a graph neural network (FoldingGATv2) to
   predict the folding demand score from sequence and structure.
3. **Codon optimisation** — integrate the trained model into the RiboDecode
   optimiser as a fold-penalty term, producing folding-aware optimised sequences.
4. **Evaluation** — test, across a panel of genes, whether the fold penalty
   places slow codons at positions that require co-translational pauses.

## Folding Demand Score

The **folding demand** score is a per-codon value in [0, 1] that quantifies how
much structural complexity must be assembled before the ribosome can safely
elongate past a given position.

**Formula:** `folding_demand = protein_need × (1 + β × (ribo_confirmation − 0.5))`

| Component | Definition |
|---|---|
| `protein_need` | Weighted combination of contact density, domain boundary proximity, secondary structure transitions, and pLDDT confidence from AlphaFold2. Captures *where structure requires a pause*. |
| `ribo_confirmation` | Smoothed ribosome occupancy from Ribo-seq, normalised by a global threshold (3× gene mean). Captures *where ribosomes actually pause in vivo*. |
| `β = 0.6` | Modulation strength: ribo_confirmation adjusts protein_need by ±30%, never zeroing it out. Structural signal dominates; Ribo-seq calibrates. |

The additive modulation (rather than multiplication) ensures that noise in
Ribo-seq data does not eliminate the structural signal at positions where real
folding demand exists. Normalisation is applied globally across all genes, so the
score is comparable between simple and complex proteins.

## GNN Architecture (FoldingGATv2)

The model is a **3-layer GATv2** (Graph Attention Network v2):

- **Node features:** 18 structural features (pLDDT, contact density, domain
  boundaries, secondary structure, RNA unpaired probabilities, opening energies,
  sliding-window paired probability) + a learned 8-dimensional codon embedding
  (64 codons + UNK).
- **Three edge types:**
  - *Sequential edges* at distances ±1, ±3, ±5 codons — capture rare-codon clusters
  - *Protein 3D contact edges* from AlphaFold2 Cα coordinates (threshold: 8 Å,
    min sequence separation: 6)
  - *RNA base-pairing edges* from RNAfold dot-bracket structure
- **Loss:** Hybrid weighted MSE + BCE: `0.7 × MSE(w·(p−t)²) + 0.3 × BCE`, with
  `w = 1 + 5·fd` to up-weight high-folding-demand positions.
- **Output:** Sigmoid activation → folding_demand prediction in [0, 1].
- **Training:** Gene-level train/val split (80/20, with wild-type and de novo
  versions of each gene kept on the same side to prevent leakage), AdamW +
  ReduceLROnPlateau, early stopping (patience=8), MPS/CUDA/CPU auto-detection.

## Codon Optimisation with Fold Penalty

The trained folding demand model is integrated into the RiboDecode optimiser as
an additional loss term. The optimiser uses activation maximisation over a
learned soft codon distribution, with a fitness function combining three terms:

| Term | Source | Goal |
|---|---|---|
| RPF loss | RiboDecode score model | Maximise predicted ribosome load |
| MFE loss | RiboDecode MFE model | Maximise mRNA structural stability |
| Fold penalty | FoldingGATv2 (this work) | Place slow codons at high-folding-demand positions |

The fold penalty is `mean(fd_pred · E[TAI])`, where `fd_pred` is refreshed from
the GNN every N optimisation steps and `E[TAI]` is the expected tRNA Adaptation
Index under the current soft codon distribution. The penalty weight is controlled
by `--fold_gamma` (0 = baseline RiboDecode, 0.3 = recommended, 0.5 = strong).

## Repository Structure

```
Master/
├── src/
│   ├── preprocessing/          Data pipeline (run in order)
│   │   ├── utils/
│   │   │   └── rnaplfold_utils.py   Shared RNAplfold parsing and aggregation
│   │   ├── filter_and_kallisto.py   Filter GENCODE transcripts; select isoform by TPM
│   │   ├── Download_gene_sequences.py   Extract per-gene cDNA FASTAs from Ensembl
│   │   ├── fast_ribo_analysis.py    Custom seed aligner → per-codon occupancy CSV
│   │   ├── run_ribo_all_genes.py    Batch wrapper over fast_ribo_analysis.py
│   │   ├── add_structure_features.py   Extract AlphaFold2 structural features per codon
│   │   ├── add_structure_to_simulated.py   Map WT structural features to GEMORNA sequences
│   │   ├── add_rna_structure.py         Run RNAplfold on WT sequences; add RNA features
│   │   ├── add_rnaplfold_to_simulated.py  Run RNAplfold on synthetic sequences
│   │   ├── add_folding_demand.py    Combine all features → folding_demand score
│   │   ├── run_gemorna_and_simulate.py   Generate de novo mRNAs + simulate ribosome counts
│   │   └── run_ablation.py          Preprocessing ablation configs
│   ├── training/
│   │   ├── train_gnn.py            FoldingGATv2 training
│   │   ├── train_gnn_rna_only.py   Ablation: RNA features only
│   │   └── run_ablation.py         Run over all edge configs; print comparison table
│   ├── optimization/
│   │   ├── ribodecode_with_fold_penalty.py   Optimiser with integrated fold penalty
│   │   ├── make_configs.py          Build per-gene sweep configs
│   │   └── sweep_array.job          Slurm array job for the HPC sweep
│   └── analysis/
│       ├── analysis_folding_demand.py   Validate folding_demand vs Ribo-seq occupancy
│       ├── analyze_predictions.py       Analyse prediction patterns
│       └── evaluate_sweep.py            Statistical evaluation of optimised sequences
├── data/
│   ├── genes/                  Per-gene/isoform FASTA files
│   ├── ribo_counts/            WT per-codon occupancy CSVs (pipeline output)
│   ├── ribo_counts_simulated/  Simulated per-codon occupancy CSVs
│   ├── alphafold_results/      AlphaFold2 predictions (PDB/CIF + PAE JSON)
│   ├── genes_gemorna/          GEMORNA-generated de novo sequences
│   ├── optimized/              Optimiser output (FASTA + JSON per gene/gamma)
│   └── results/                Training metrics and trained model checkpoints
├── RiboDecode/                 RiboDecode optimiser (see Installation)
├── GEMORNA/                    GEMORNA generative model (see Installation)
├── isoform_selection.json      Master gene → canonical isoform index
└── .gitignore
```

## Installation

### Core environment

```bash
# Create and activate the project environment
conda create -n folding python=3.12
conda activate folding

# Core dependencies
pip install torch torch-geometric biopython pandas numpy scipy matplotlib
pip install scikit-learn requests

# RNA structure (ViennaRNA)
conda install -c bioconda viennarna

# Kallisto (for isoform quantification)
conda install -c bioconda kallisto
```

### AlphaFold2 (local ColabFold)

```bash
# Install LocalColabFold following the upstream instructions:
#   https://github.com/YoshitakaMo/localcolabfold
# Predictions are written to data/alphafold_results/<GENE>/ as PDB + PAE JSON.
```

### RiboDecode

RiboDecode is the base codon optimiser that this project extends. Clone it into
the project root so that the fold-penalty optimiser can import its modules.

```bash
# Clone RiboDecode into the project root
git clone https://github.com/zhangtaolab/RiboDecode.git
# (adjust the URL to the actual RiboDecode repository)

# RiboDecode ships pretrained score and MFE models; place them under
#   RiboDecode/score_model/best_model.p
#   RiboDecode/score_model/model_config.json
# following the RiboDecode README.

# The optimiser imports RiboDecode.dataset, RiboDecode.models, and
# RiboDecode.score_model, so RiboDecode must be importable from the project root.
```

### GEMORNA

GEMORNA generates the de novo sequences used to augment the training set. It runs
in its own environment because its dependencies differ from the core environment.

```bash
# Clone GEMORNA into the project root
git clone https://github.com/<gemorna-repo>/GEMORNA.git
cd GEMORNA

# Create a dedicated environment
conda create -n gemorna python=3.10
conda activate gemorna
pip install -r requirements.txt

# Download the pretrained checkpoints into GEMORNA/checkpoints/:
#   gemorna_cds.pt, gemorna_5utr.pt, gemorna_3utr.pt
# following the GEMORNA README.
cd ..
```

## Pipeline Execution

### 1. Preprocessing

```bash
conda activate folding

# Select canonical isoforms (requires GENCODE FASTA + Ribo-seq FASTQ)
python src/preprocessing/filter_and_kallisto.py \
    --gencode_fa Homo_sapiens.GRCh38.cdna.all.fa \
    --fastq data/ribo_fastq/SRR10072555.fastq

# Extract per-gene cDNA sequences
python src/preprocessing/Download_gene_sequences.py Homo_sapiens.GRCh38.cdna.all.fa

# Align ribosome footprints → per-codon occupancy (batch over all genes)
python src/preprocessing/run_ribo_all_genes.py

# Run AlphaFold2 + extract structural features per codon
python src/preprocessing/add_structure_features.py

# Compute RNA secondary structure features (WT)
python src/preprocessing/add_rna_structure.py

# Compute folding demand score
python src/preprocessing/add_folding_demand.py
```

### 2. De novo training data (GEMORNA + simulation)

```bash
# Generate GEMORNA de novo sequences + simulate ribosome counts (sTASEP)
conda activate gemorna
python src/preprocessing/run_gemorna_and_simulate.py

# Add features to the de novo sequences
conda activate folding
python src/preprocessing/add_structure_to_simulated.py
python src/preprocessing/add_rnaplfold_to_simulated.py
python src/preprocessing/add_folding_demand.py   # processes both WT and de novo dirs
```

### 3. Train the folding demand model

```bash
# Train FoldingGATv2 (full edge config)
python src/training/train_gnn.py --edges seq+prot+rna

# Run the edge-config ablation and print the comparison table
python src/training/run_ablation.py
```

### 4. Codon optimisation

```bash
# Single gene, single fold-penalty weight
python src/optimization/ribodecode_with_fold_penalty.py \
    --cds ACTB \
    --cds_seq <full_mRNA_sequence> \
    --env HEK293T \
    --mfe_weight 0.7 \
    --optim_epoch 5 \
    --alpha 100 --beta 100 \
    --fold_gamma 0.3 \
    --gnn_update_every 250

# Full sweep across genes and gamma values (HPC, Slurm)
python src/optimization/make_configs.py
sbatch src/optimization/sweep_array.job
```

### 5. Evaluation

```bash
# Validate folding_demand against Ribo-seq occupancy
python src/analysis/analysis_folding_demand.py

# Statistical evaluation of the optimised sequences across all genes
python src/analysis/evaluate_sweep.py
```

## Data

| Source | Description | Size |
|---|---|---|
| Ribo-seq | SRR10072555 (human HEK293, ribosome profiling) | 5.8 GB FASTQ |
| AlphaFold2 | Local ColabFold predictions | 2.3 GB |
| RNAplfold | ViennaRNA local structure (computed from FASTA) | — |
| GENCODE | `Homo_sapiens.GRCh38.cdna.all.fa` (Ensembl cDNA reference) | 800 MB |
| GEMORNA | De novo mRNA sequences (generative model, Li et al. 2025) | 1.5 MB |

## Development History

The folding demand model went through five iterations during development. The
current model corresponds to the final iteration; earlier versions are kept for
reference only.

| Iteration | Key change |
|---|---|
| 1 | Baseline: GATv2, uniform MSE, WT data only |
| 2 | Weighted MSE (`w = 1 + 3·fd`), WT only |
| 3 | Sigmoid output, hybrid MSE/BCE, WT + simulated data |
| 4 | 3-layer GATv2, codon + amino acid features |
| 5 (current) | Learned codon embeddings (8-dim), extended seq edges at ±1/3/5 codons |

## Status

MSc thesis in progress — ITU Copenhagen, 2026.