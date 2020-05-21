import time
import pandas as pd
import os
import pickle
import torch

from tfn import LOG_FILE, SAVED_MODELS_DIR


def log_sk_model(model, mean, std, params):
    if not os.path.exists(SAVED_MODELS_DIR):
        os.mkdir(SAVED_MODELS_DIR)
    model_name = type(model).__name__
    # Save model instance
    file_name = f'{model_name}-{int(time.time())}.pkl'
    with open(os.path.join(SAVED_MODELS_DIR, file_name), 'wb') as f:
        pickle.dump(model, f)

    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
    else:
        log_df = pd.DataFrame(columns=['model', 'file', 'mean_result', 'std_result'])

    row_dict = {
        'model': model_name,
        'file': file_name,
        'mean_result': mean,
        'std_result': std,
        **params
    }

    for k in row_dict:
        if k not in log_df.columns:
            log_df[k] = ""

    new_df = pd.DataFrame([row_dict], columns=row_dict.keys())
    log_df = log_df.append(new_df, sort=False)

    log_df.to_csv(LOG_FILE, index=False)

def log_torch_model(model, acc, params, std=None):
    if not os.path.exists(SAVED_MODELS_DIR):
        os.mkdir(SAVED_MODELS_DIR)
    model_name = type(model).__name__
    # Save model instance
    file_name = f'{model_name}-{int(time.time())}.pkl'
    # Save model
    torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, file_name))

    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
    else:
        log_df = pd.DataFrame(columns=['model', 'file', 'mean_result', 'std_result'])

    row_dict = {
        'model': model_name,
        'file': file_name,
        'mean_result': acc,
        **params
    }
    if std:
        row_dict['std_result'] = std

    for k in row_dict:
        if k not in log_df.columns:
            log_df[k] = ""

    new_df = pd.DataFrame([row_dict], columns=row_dict.keys())
    log_df = log_df.append(new_df, sort=False)

    log_df.to_csv(LOG_FILE, index=False)

