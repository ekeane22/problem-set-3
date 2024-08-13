'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

logging.basicConfig(
    filename='transform.log',
    filemode='w',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''
    try: 
        macro_precision_list = []
        macro_recall_list = []
        macro_f1_list = []
    
        for genre in genre_list: 
            tp = genre_tp_counts.get(genre, 0)
            fp = genre_fp_counts.get(genre,0)
            fn = genre_true_counts.get(genre, 0) - tp 
    
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0 
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0 
            f1 = 2 * (precision * recall) / (precision + recall)
            
            macro_precision_list.append(precision)
            macro_recall_list.append(recall)
            macro_f1_list.append(f1)
        
        #micro metrics 
        true_total = sum(genre_true_counts.values())
        tp_total = sum(genre_tp_counts.values())
        fp_total = sum(genre_fp_counts.values())
        fn_total = true_total = tp_total 
        
        





    true = [genre_true_counts[genre] for genre in genre_list]
    tp1 = [genre_tp_counts[genre] for genre in genre_list]
    fp1 = [genre_fp_counts[genre] for genre in genre_list]
    fn1 = [true[genre_list.index(genre)] - genre_tp_counts[genre] for genre in genre_list]

    true_mic = sum(true)
    tp_mic = sum(tp1)
    fp_mic = sum(fp1)
    fn_mic = sum(fn1)


    micro_precision = 
    macro_precision = 
    micro_recall = 
    macro_recall = 
    micro_f1 = 
    macro_f1 = 
    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

    # Your code here
