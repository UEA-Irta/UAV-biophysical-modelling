
# RF_processing

RF_params = {
    'n_estimators': 100,
    'min_samples_split': 2,
    'criterion': 'squared_error',
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'random_state': 42
}


RF_predict = {
    "dataset": "example_dataset_predict.xlsx",
    "features": ['fc','hc',	'hc_mean', 'NDVI_mean', 'NDVI_sd', 'MTVI2_mean', 'MTVI2_sd', 'SAVI_mean', 'SAVI_sd'],
    "target": 'LAI',
    "normalize": True,
    "min_max_values": None
}
