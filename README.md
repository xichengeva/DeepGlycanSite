# DeepGlycanSite: A Highly Accurate Glycan-binding Site Prediction Algorithm

Implementation of DeepGlycanSite, a state-of-the-art method for glycan-binding site prediction, by Xinheng He ..., Dingyan Wang, and Xi Cheng.
This repository contains all code, instructions, and model weights necessary to run the method or to retrain a model. 
If you have any questions, feel free to open an issue or reach out to us: [wangdy@lglab.ac.cn](wangdy@lglab.ac.cn), [xicheng@simm.ac.cn](xicheng@simm.ac.cn)

![Alt Text](visualizations/overview.png)


# Dataset

If you want to train one of our models with the data then: 
1. download pdb ids from dataset/
2. divide them to ligands and proteins, like 
2. unzip the directory and place it into `data` such that you have the path `data/PDBBind_processed`


# Setup Environment

We will set up the environment using [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). Clone the current repo

    git clone https://github.com/xichengeva/DeepGlycanSite.git

This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):

    conda create --name DeepGlycanSite python=3.9
    conda activate DeepGlycanSite

For CPU:
    pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
    
    pip install torch_geometric
    
    pip install "fair-esm[esmfold]"
    
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
    
    pip install pytorch-lightning &&    pip install h5py &&    pip install  ase && pip install rdkit && pip install jupyter && pip install Bio && pip install pandas &&  pip install MDAnalysis

For GPU:
    pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
    
    pip install torch-geometric
    
    pip install "fair-esm[esmfold]"
    
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu116.html
    
    pip install pytorch-lightning &&    pip install h5py &&    pip install  ase && pip install rdkit && pip install jupyter && pip install Bio && pip install pandas &&  pip install MDAnalysis

You should also download the checkpoints from https://huggingface.co/Xinheng/DeepGlycanSite/tree/main, just put rec_only.ckpt and with_ligand.ckpt to ./ckpt, and mol_pre_all_h_220816.pt to ./src/unimol_tools/weights

# Running DeepGlycanSite on test system 

The protein inputs need to be `.pdb` files. The ligand input need to be `.sdf` files. Both the files need HYDROGENS to be added.

Run inference with only receptor:

    python single_case_prediction.py --conf P2Y14_example/hparams_rec.yaml  --ckpt_path ckpts/rec_only.ckpt --input_fn P2Y14_example/P2Y14_AF.pdb  --out_path P2Y14_example/ --output_fn P2Y14_af.txt

P2Y14_example/P2Y14_af.txt will be the output file and the second column is the probability for prediction

Run inference with receptor and ligand:

    python single_case_prediction.py --conf hparams.yaml  --ckpt_path ckpts/with_ligand.ckpt --input_fn P2Y14_example/P2Y14_AF.pdb,P2Y14_example/GDP.sdf --out_path P2Y14_example/ --output_fn P2Y14_af_GDP.txt

Use comma to connect pdb and ligand sdf for activate DeepGlycanSite+ligand model, P2Y14_example/P2Y14_af_GDP.txt will be the output file

# Retraining DeepGlycanSite
Use single_case_prediction.py for per protein and ligands, then combine the .pt files in out_path, and you can train it with DeepGlycanSite.py or DeepGlycanSite_lig.py with following codes:

    python DeepGlycanSite.py --conf P2Y14_example/hparams_rec.yaml --output-path DeepGlycanSite_rec
    
    python DeepGlycanSite_lig.py --conf P2Y14_example/hparams.yaml --output-path DeepGlycanSite_lig

# License
MIT

# Acknowledgements

We thank Hualiang Jiang and Huaqiang Eric Xu for guiding us such a work.

