# Predicting Translational Pause Sites in mRNA Using Graph Neural Networks

## Biological Problem

During translation, ribosomes do not synthesize proteins at a uniform rate. They pause at specific codons, and growing evidence suggests these pauses are functionally important: they give nascent protein domains time to fold co-translationally before they emerge from the ribosome tunnel. This co-translational folding is especially critical for proteins with complex multi-domain architectures, where premature folding of one domain in the presence of another can lead to aggregation or misfolding. Beyond basic biology, this has direct relevance to mRNA vaccine and therapeutic protein design — the codon usage of a synthetic mRNA determines ribosome dynamics, which in turn affects protein yield and correct folding. This project develops a machine-learning pipeline to predict *where* ribosomes should pause, given only the protein structure and the mRNA sequence.

## Folding Demand Score

The **folding demand** score is a per-codon value in [0, 1] that quantifies how much structural complexity must be assembled before the ribosome can safely elongate past a given position.

**Formula:** `folding_demand = protein_need × (1 + β × (ribo_confirmation − 0.5))`

| Component | Definition |
|---|---|
| `protein_need` | Weighted combination of contact density, domain boundary proximity, secondary structure transitions, and pLDDT confidence from AlphaFold2. Captures *where structure requires a pause*. |
| `ribo_confirmation` | Smoothed ribosome occupancy from Ribo-seq, normalised by a global threshold (3× gene mean). Captures *where ribosomes actually pause in vivo*. |
| `β = 0.6` | Modulation strength: ribo_confirmation adjusts protein_need by ±30%, never zeroing it out. Structural signal dominates; Ribo-seq calibrates. |

The additive modulation (rather than multiplication) ensures that noise in Ribo-seq data does not eliminate the structural signal at positions where real folding demand exists.

## GNN Architecture

The model is a **3-layer GATv2** (Graph Attention Network v2) with:

- **Node features:** 18 structural features (pLDDT, contact density, domain boundaries, secondary structure, RNA unpaired probabilities, opening energies, sliding-window paired_prob) + learned 8-dimensional codon embedding (64 codons + UNK)
- **Three edge types:**
  - *Sequential edges* at distances ±1, ±3, ±5 codons — capture rare-codon clusters
  - *Protein 3D contact edges* from AlphaFold2 Cα coordinates (threshold: 8 Å, min sequence separation: 6)
  - *RNA base-pairing edges* from RNAfold dot-bracket structure
- **Loss:** Hybrid weighted MSE + BCE: `0.7 × MSE(w·(p−t)²) + 0.3 × BCE`, with `w = 1 + 5·fd` to up-weight high-folding-demand positions
- **Output:** Sigmoid activation → folding_demand prediction in [0, 1]
- **Training:** Gene-level train/val split (80/20), AdamW + ReduceLROnPlateau, early stopping (patience=8), MPS/CUDA/CPU auto-detection

## Repository Structure

```
Master/
├── src/
│   ├── preprocessing/          Data pipeline (run in order)
│   │   ├── utils/
│   │   │   └── rnaplfold_utils.py   Shared RNAplfold parsing and aggregation
│   │   ├── filter_and_kallisto.py   Filter GENCODE transcripts; select canonical isoform by TPM
│   │   ├── Download_gene_sequences.py   Extract per-gene cDNA FASTAs from Ensembl reference
│   │   ├── fast_ribo_analysis.py    k-mer ribosome footprint alignment → per-codon occupancy CSV
│   │   ├── run_ribo_all_genes.py    Batch wrapper over fast_ribo_analysis.py
│   │   ├── add_structure_features.py   Extract AlphaFold2 structural features per codon
│   │   ├── add_structure_to_simulated.py   Map WT structural features to GEMORNA sequences
│   │   ├── add_rna_structure.py         Run RNAplfold on WT sequences; add RNA features
│   │   ├── add_rnaplfold_to_simulated.py  Run RNAplfold on synthetic sequences
│   │   ├── add_folding_demand.py    Combine all features → folding_demand score (v5)
│   │   ├── run_gemorna_and_simulate.py   Generate synthetic mRNAs + simulate ribosome counts
│   │   └── run_ablation.py          Preprocessing ablation configs
│   ├── training/
│   │   ├── train_gnn_v5.py         Current production GNN (v5)
│   │   ├── train_gnn_rna_only.py   Ablation: RNA features only
│   │   └── run_ablation.py         Run v5 over all edge configs; print comparison table
│   └── analysis/
│       ├── analysis_folding_demand.py   Validate folding_demand vs actual Ribo-seq occupancy
│       └── analyze_predictions.py       Analyse prediction patterns and target distribution
├── _archive/                   v1–v4 training scripts (superseded by v5)
├── data/
│   ├── genes/                  Per-gene/isoform FASTA files
│   ├── ribo_counts/            WT per-codon occupancy CSVs (pipeline output)
│   ├── ribo_counts_simulated/  Simulated per-codon occupancy CSVs
│   ├── alphafold_results/      AlphaFold2 predictions (PDB/CIF + PAE JSON)
│   └── results/                Training metrics JSON per version and edge config
├── isoform_selection.json      Master gene → canonical isoform index
├── check_stop_codons.py        Data validation: verify stop codon positions
└── .gitignore
```

## Pipeline Execution Order

```
# 1. Select canonical isoforms (requires GENCODE FASTA + Ribo-seq FASTQ)
python src/preprocessing/filter_and_kallisto.py \
    --gencode_fa Homo_sapiens.GRCh38.cdna.all.fa \
    --fastq data/ribo_fastq/SRR10072555.fastq

# 2. Extract per-gene cDNA sequences
python src/preprocessing/Download_gene_sequences.py Homo_sapiens.GRCh38.cdna.all.fa

# 3. Align ribosome footprints → per-codon occupancy (runs all genes in batch)
python src/preprocessing/run_ribo_all_genes.py

# 4. Run AlphaFold2 + extract structural features per codon
python src/preprocessing/add_structure_features.py

# 5. Compute RNA secondary structure features (WT)
python src/preprocessing/add_rna_structure.py

# 6. Compute folding demand score
python src/preprocessing/add_folding_demand.py

# --- Optional: generate synthetic training data ---
# 7. Generate GEMORNA-optimised sequences + simulate ribosome counts
python src/preprocessing/run_gemorna_and_simulate.py

# 8. Add structural features to simulated sequences
python src/preprocessing/add_structure_to_simulated.py
python src/preprocessing/add_rnaplfold_to_simulated.py
python src/preprocessing/add_folding_demand.py   # processes both dirs

# --- Training ---
# 9. Train v5 GNN (full edge config)
python src/training/train_gnn_v5.py --edges seq+prot+rna

# 10. Run ablation over all edge configs and compare
python src/training/run_ablation.py

# --- Analysis ---
# 11. Validate folding_demand against Ribo-seq
python src/analysis/analysis_folding_demand.py

# 12. Analyse prediction patterns
python src/analysis/analyze_predictions.py
```

## Requirements

```bash
# Create and activate the project environment
conda create -n folding python=3.12
conda activate folding

# Core dependencies
pip install torch torch-geometric biopython pandas numpy scipy matplotlib
pip install scikit-learn requests

# RNA structure (ViennaRNA)
conda install -c bioconda viennarna

# AlphaFold2 (local ColabFold)
# See localcolabfold/ — follow ColabFold installation instructions

# Kallisto (for isoform quantification)
conda install -c bioconda kallisto
```

## Data

| Source | Description | Size |
|---|---|---|
| Ribo-seq | SRR10072555 (human HEK293, ribosome profiling) | 5.8 GB FASTQ |
| AlphaFold2 | Local ColabFold predictions for 97 genes | 2.3 GB |
| RNAplfold | ViennaRNA local structure (computed from FASTA) | — |
| GENCODE | `Homo_sapiens.GRCh38.cdna.all.fa` (Ensembl cDNA reference) | 800 MB |
| GEMORNA | Synthetic mRNA sequences (generative model, Zhang et al. 2025) | 1.5 MB |

## Model Status

> **v5 is the current production model.** v1–v4 are archived in `_archive/` for reference.

| Version | Key change |
|---|---|
| v1 | Baseline: GATv2, uniform MSE, WT data only |
| v2 | Weighted MSE (`w = 1 + 3·fd`), WT only |
| v3 | Sigmoid output, hybrid MSE/BCE, WT + simulated data |
| v4 | 3-layer GATv2, codon + amino acid features |
| **v5** | **Learned codon embeddings (8-dim), extended seq edges at ±1/3/5 codons** |

## Status

MSc thesis in progress — ITU Copenhagen, 2026.
