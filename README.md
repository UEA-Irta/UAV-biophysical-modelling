
# UAV Biophysical Traits Estimation

## Objective

This library is designed to estimate biophysical variables such as LAI (Leaf Area Index), fIPAR (Fraction of Incoming Photosynthetically Active Radiation), and other related traits derived from UAV data using configurable empirical models, specifically Random Forest (RF) and Neural Networks (NN).

The library provides:

1. **Data Preprocessing**: Processes UAV-derived datasets, ensuring compatibility for model training and inference.
2. **Model Training and Testing**: Includes tools to configure, train, and validate RF and NN models for biophysical variable estimation.
3. **Inference**: Applies trained models to predict biophysical variables on new datasets.

---

## Installation

To use this project, ensure you have Python 3.x installed. Then, clone the repository and install the required dependencies.

1. Clone the repository:
   ```bash
   git clone https://github.com/UEA-Irta/UAV-biophysical-estimation.git


2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   conda env --create -f uav-biophysical-estimation.yml
   conda env update -f uav-biophysical-estimation.yml


## Requirements

This project requires the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `keras`
- `joblib`
- `geopandas`
- `rasterio`
- `fiona`
- `shapely`
- `rasterstats`
- `scikit-image`

---

## DEMO


In the `utils` folder, there are two Excel files provided for testing purposes:

1. `example_dataset_train_val_test.xlsx`: This file is used to create the data splits (training, validation, and testing) and train the model.

2. `example_dataset_predict.xlsx`: This file evaluates the model's performance on a separate dataset.


## Main Scientific References

- C.Minuesa, M. Quintanilla-Albornoz, J. Gené-Mola, A. Pelechá, L. Aparicio, J.Bellvert. 2025. Machine learning for leaf area index estimation in almond orchards using UAV multispectral and point-cloud data. 15th European Conference on Precision Agriculture - ECPA'25. Barcelona, Spain. 29 june - 3 july 2025. Submitted
- Gao, R., Torres-Rua, A., Aboutalebi, M., White, W., Anderson, M., Kustas, W., Agam, N., Alsina, M., Alfieri, J., Hipps, L., Dokoozlian, N., Nieto, H., Gao, F., McKee, L., Prueger, J., Sanchez, L., Mcelrone, A., Bambach-Ortiz, N., Coopmans, C., and Gowing, I.: LAI estimation across California vineyards using sUAS multi-seasonal multi-spectral, thermal, and elevation information and machine learning, Irrigation Sci., 40, 731–759, https://doi.org/10.1007/s00271-022-00776-0, 2022. 
- Quintanilla-Albornoz, M., Miarnau, X., Pelechá, A., Casadesús, J., García-Tejera, O., and Bellvert, J.: Evaluation of transpiration in different almond production systems with two-source energy balance models from UAV thermal and multispectral imagery, Irrigation Sci., https://doi.org/10.1007/s00271-023-00888-1, 2023. 

## Notes

- **Extensibility**: The pipeline is designed to support additional features or biophysical variables by updating the configuration file and adapting the utility scripts.
- **Reproducibility**: Using the `config/` directory ensures consistency in parameter settings across different runs.

---

## Contact

For issues or feature requests, please contact: [cesar.minuesa@irta.cat].

