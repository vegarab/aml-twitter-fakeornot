import __main__
from datetime import datetime
import csv
import os


def export_results(acc, roc, f1):
    results_file = '../data/results.csv'
    model = os.path.basename(__main__.__file__)
    dt = datetime.now()
    fields = [model, dt, acc, roc, f1]
    with open(results_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)