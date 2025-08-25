
# splits_processing

splits_params = {
    "dataset_path": "example_dataset_train_val_test.xlsx",
    "features": ['fc','hc',	'hc_mean', 'NDVI_mean', 'NDVI_sd', 'MTVI2_mean', 'MTVI2_sd', 'SAVI_mean', 'SAVI_sd'],
    "target": 'LAI',
    "trials": {
        'trial1': 'plot == "CLD"',
        'trial2': 'plot == "BRB"',
    },
    "training_size": 0.7,
    "test_size": 0.15,
    "normalize": True,
    "output_path": "output_splits"
}