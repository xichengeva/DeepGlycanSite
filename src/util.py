import os
import sys
import random
import datetime
import numpy as np
import argparse
import yaml
import torch
import requests  
from tqdm import tqdm  

class LoadFromFile(argparse.Action):
    # parser.add_argument('--file', type=open, action=LoadFromFile)
    def __call__(self, parser, namespace, values, option_string=None):
        if values.name.endswith("yaml") or values.name.endswith("yml"):
            with values as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            if 'v2s_update' in config.keys():
                del config['v2s_update']
            for key in config.keys():
                if key not in namespace:
                    raise ValueError(f"Unknown argument in config file: {key}")
            namespace.__dict__.update(config)
        else:
            raise ValueError("Configuration file must end with yaml or yml")


def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else:
        time = None
        line = string

    print(line, file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_args(args):
    """ sanity check for arguments """
    if args["checkpoint"] is not None and not os.path.exists(args["checkpoint"]):
            sys.exit("checkpoint [%s] does not exists" % (args["checkpoint"]))


def set_output(args, string):
    """ set output configurations """
    output, save_prefix = sys.stdout, None
    if args["output_path"] is not None:
        save_prefix = args["output_path"]
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(args["output_path"] + "/" + string + ".txt", "a")

    return output, save_prefix


  
def auto_download(type):   
    os.makedirs('ckpts',exist_ok=True)
    def download_file(url, filename):  
        print(f'downloading {filename}...')
        response = requests.get(url, stream=True)  
        total_size_in_bytes= int(response.headers.get('content-length', 0))  
        block_size = 1024 #1 Kibibyte  
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)  
        with open(filename, 'wb') as file:  
            for data in response.iter_content(block_size):  
                progress_bar.update(len(data))  
                file.write(data)  
        progress_bar.close()  
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:  
            print("ERROR, something went wrong")  
  
    if type == 'rec':    
        if not os.path.exists('ckpts/rec_only.ckpt'):    
            url = 'https://huggingface.co/Xinheng/DeepGlycanSite/blob/main/rec_only.ckpt'  
            alt_url = 'https://zenodo.org/records/10065607/files/rec_only.ckpt'  
            try:  
                download_file(url, 'ckpts/rec_only.ckpt')  
            except:  
                download_file(alt_url, 'ckpts/rec_only.ckpt')  
  
    elif type == 'lig':    
        if not os.path.exists('ckpts/with_ligand.ckpt'):    
            url = 'https://huggingface.co/Xinheng/DeepGlycanSite/blob/main/with_ligand.ckpt'  
            alt_url = 'https://zenodo.org/records/10065607/files/with_ligand.ckpt'  
            try:  
                download_file(url, 'ckpts/with_ligand.ckpt')  
            except:  
                download_file(alt_url, 'ckpts/with_ligand.ckpt')  
  
        if not os.path.exists('src/unimol_tools/weights/mol_pre_all_h_220816.pt'):    
            url = 'https://huggingface.co/Xinheng/DeepGlycanSite/blob/main/mol_pre_all_h_220816.pt'  
            alt_url = 'https://zenodo.org/records/10065607/files/mol_pre_all_h_220816.pt'  
            # try:  
            #     download_file(url, 'src/unimol_tools/weights/mol_pre_all_h_220816.pt')  
            # except:  
            download_file(alt_url, 'src/unimol_tools/weights/mol_pre_all_h_220816.pt')  
