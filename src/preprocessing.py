'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
   - Further info and hints are provided in the docstrings
   - These should return values when called by the main.py
'''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_data():
    '''
    Load data from CSV files
  
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    model_pred_df = pd.read_csv('../data/prediction_model_03.csv')
    genres_df = pd.read_csv('../data/genres.csv')

    return model_pred_df, genres_df
    
def process_data(model_pred_df, genres_df):
    '''
    Process data to clean column names, add context columns, encode genres, 
    and calculate genre statistics
    More specifically: 
    - The function renames the column names and removes 'tt' from the imbd_id column. 
    - Created new context columns. 
    - Applying one-hot encoding to encode the categorical features. 
    - calculated the returns. 
    
    Args: 
        model_pred_df (DF): DataFrame containing model predictions with columns 'imdb_id', 'predicted', 'actual_genre', and 'correct'.
        genres_df (DF): DataFrame containing genre information with one column, called 'genre' listing all possible genres.
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    #clean the column names 
    model_pred_df.rename(columns={
        'correct?': 'correct',
        'actual genres': 'actual_genre'
    }, inplace=True)
    print("Column names after renaming:", model_pred_df.columns.tolist())
    
    model_pred_df['imdb_id'] = model_pred_df['imdb_id'].str.replace('tt', '', regex=False)
    
# add context to the columns and create new columns - part of feature selection 
    model_pred_df['actual_predicted'] = model_pred_df['actual_genre'].astype(str) + '_' + model_pred_df['predicted']
    model_pred_df['correct_actual'] = model_pred_df['correct'].astype(str) + '_' + model_pred_df['actual_genre'].astype(str)
    model_pred_df['ID_predicted'] = model_pred_df['imdb_id'].astype(str) + '_' + model_pred_df['predicted']

#one hot encoding - from the lecture 
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(model_pred_df[['predicted', 'actual_predicted', 'correct_actual', 'ID_predicted']]).toarray()
    encoded_feature_names = encoder.get_feature_names_out(['predicted', 'actual_predicted', 'correct_actual', 'ID_predicted'])
    one_hot_encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names)
    one_hot_encoded_df['imdb_id'] = model_pred_df['imdb_id']
    
    genre_list = genres_df['genre'].tolist()
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    
    
    if model_pred_df['actual_genre'].isnull().any():
        print("Warning: NaN/Null values found in 'actual_genre' column")

    for index, row in model_pred_df.iterrows(): 
        try: 
            actual_genres = eval(row['actual_genre']) if isinstance(row['actual_genre'], str) else row['actual_genre']
            
            if not actual_genres or (isinstance(actual_genres, list) and all(genre == '' for genre in actual_genres)):
                continue 
            
            if row['correct'] == 1: 
                for genre in actual_genres:
                    genre_true_counts[genre] += 1
                    if row['predicted'] == genre: 
                        genre_tp_counts[genre] += 1
            else:
                for genre in actual_genres:
                    if row['predicted'] != genre: 
                        genre_fp_counts[genre] += 1
        except Exception as row_error:
            print(f"Error processing row {index}: {row_error}")
            print(f"Row data: {row}")
                    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
                    
if __name__ == "__main__": 
    try: 
        model_pred_df, genres_df = load_data()
        
        genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts = process_data(model_pred_df, genres_df)
        
        print("Genre List:", genre_list)
        print("Genre True Counts:", genre_true_counts)
        print("Genre True Positive Counts:", genre_tp_counts)
        print("Genre False Positive Counts:", genre_fp_counts)
    except Exception as e: 
        print(f"An error occured: {e}")