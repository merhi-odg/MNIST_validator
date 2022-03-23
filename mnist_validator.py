import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# modelop.metrics
def metrics(dataframe) -> dict:

    # Basic stats
    num_records = dataframe.shape[0]
    matching_rows = dataframe[dataframe["score_tf"] == dataframe["score_sklearn"]]
    percent_match = np.round(matching_rows.shape[0] / num_records * 100, 2)

    # Confusion Matrix
    labels_sorted = sorted(pd.Series(dataframe["score_tf"]).unique())
    conf_mat = confusion_matrix(
        y_true=dataframe["score_tf"],
        y_pred=dataframe["score_sklearn"],
        normalize="all",
        labels=labels_sorted,
    ).round(4)

    label_strings = [str(label) for label in labels_sorted]

    # conf_mat is a numpy array. We turn it into array of dicts
    conf_mat_json = []
    for idx, _ in enumerate(labels_sorted):
        conf_mat_json.append(dict(zip(label_strings, conf_mat[idx, :].tolist())))

    # Output
    results = {
        "percent_mismatch": 100 - percent_match,
        "performance": [
            {
                "test_name": "Output Comparison",
                "test_category": "performance",
                "test_type": "comparison",
                "test_id": "output_comparison",
                "values": {
                    "record_count": num_records,
                    "pecent_match": percent_match,
                    "percent_mismatch": 100 - percent_match,
                    "confusion_matrix": conf_mat_json,
                },
            }
        ]
    }
    yield results
