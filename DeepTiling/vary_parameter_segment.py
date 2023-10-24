from glob import glob
from utils import read_file
import pandas as pd
import json
from tqdm import tqdm
import segment
import argparse
import sys
import copy
import warnings
warnings.filterwarnings("ignore")

def words_in_segment(directory):
    segment_word_count = []
    for file in glob(directory+"/*.txt"):
        txt = read_file(file)
        segment_word_count.append(len(txt.split(" ")))
    return segment_word_count

def segment_with_wd_th(wd, th):
    args.window_value = wd
    args.threshold_multiplier = th
    args.data_directory = "data/textsamples"
    segment.main(args)
    print("#"*10, "WD :" + str(wd), "TH :"+str(th), "#"*10)
    return True


class IterateParameter:
    def __init__(self, config_path) -> None:
        self.config = json.load(open(config_path))
        self.df_collector = []
        self.n_segments = None
    
    def run(self, save=None):
        for wd, th in tqdm(zip(self.config["wd"], self.config["th"]), total = len(self.config["wd"]), desc="Segment stats"):
            if segment_with_wd_th(wd, th):
                self.collect_df(wd, th)
            else:
                raise Exception("Unknown Error")
        df = pd.concat(self.df_collector, axis=0)
        if save is None:
            save = "results/segment_describe_vary_wd_th.csv"
        df.to_csv(save, index=False)
    
    def collect_df(self, wd, th):
        df = pd.DataFrame({"length":words_in_segment("results/segments")})#481
        self.n_segments = df.shape[0]
        df_less_50 = df[df["length"]<50]
        self.df_describe(df_less_50, wd, th, 'length<50')
        df_gr_100 = df[df["length"]>100]
        self.df_describe(df_gr_100, wd, th, 'length>100')
        self.df_describe(df, wd, th, 'all')
    
    def df_describe(self,df,  wd, th, name):
        df = df.describe().T
        df["wd"] = wd
        df["th"] = th
        df["operation"] = name
        df["total_segments"] = self.n_segments
        self.df_collector.append(df)



if __name__=='__main__':
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
            
    
    parser = MyParser(
            description = 'Run segmentation with parameters defined in the relative json file')
    
    parser.add_argument('--data_directory', '-data', type=str,
                        help='directory containing the data to be segmented')
    
    parser.add_argument('--config_file', '-cfg', default='./DeepTiling/parameters.json', type=str, 
                        help='Configuration file defining the hyperparameters and options to be used in training.')
    
    parser.add_argument('--out_directory', '-od', default='results', type=str,
                        help='the directory where to store the segmented texts')
    
    parser.add_argument('--window_value', '-wd', 
                        type=int,
                        default=None, 
                        help='Window value for the TextTiling algorithm, if not specified the programme will assume that the optimal value is stored in best_parameters.json file, previously obtained by running fit.py')
    
    parser.add_argument('--threshold_multiplier', '-th',
                        type=float,
                        default=None,
                        help='Threshold multiplier for the TextTiling algorithm without known number of segments, if not specified the programme will assume that the optimal value is stored in best_parameters.json file, previously obtained by running fit.py')
    
    parser.add_argument('--number_of_segments', '-ns',
                        type=int,
                        nargs = '+',
                        default=None,
                        help='List of number of segments (per document) to be returned (if known). Default is when number of segments are not known, otherwise the algorithm returns the n number of segments with higher depth score, as specified by the number at the index of the list relative to the current document.')
    
    parser.add_argument('--encoder', '-enc', type=str,
                        default=None, help='sentence encoder to be used (all sentence encoders from sentence_transformers library are supported)')
    
    parser.add_argument('--Concatenate', '-cat', type=str,
                        default=None, help='whether to concatenate the input files or to segment them individually')
                 
    parser.add_argument('--verbose', '-vb', type=bool, default=True, help='Whether to print messages during running.')
    
    
    args = parser.parse_args()
    
    iter_parameter = IterateParameter("DeepTiling/stat_parameter.json")
    iter_parameter.run()
    print("Done!")
