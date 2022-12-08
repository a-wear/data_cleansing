<br />
<p align="center"> 
  <h3 align="center">Data Cleansing for Indoor Positioning Wi-Fi Fingerprinting Datasets</h3>
</p>

[![pub package](https://img.shields.io/badge/license-CC%20By%204.0-green)]()

<!-- ABOUT THE PROJECT -->
## Abstract

Wearable and IoT devices requiring positioning and localisation services grow in number exponentially every year. This rapid growth also produces millions of data entries that need to be pre-processed prior to being used in any indoor positioning system to ensure the data quality and provide a high Quality of Service (QoS) to the end-user. In this paper, we offer a novel and straightforward data cleansing algorithm for WLAN fingerprinting radio maps. This algorithm is based on the correlation among fingerprints using the Received Signal Strength (RSS) values and the Access Points (APs)'s identifier. We use those to compute the correlation among all samples in the dataset and remove fingerprints with low level of correlation from the dataset. We evaluated the proposed method on 14 independent publicly-available datasets. As a result, an average of 14% of fingerprints were removed from the datasets. The 2D positioning error was reduced by 2.7% and 3D positioning error by 5.3% with a slight increase in the floor hit rate by 1.2% on average. Consequently, the average speed of position prediction was also increased by 14%.


```
@INPROCEEDINGS{9861169,
  author={Quezada-Gaibor, Darwin and Klus, Lucie and Torres-Sospedra, Joaquín and Lohan, Elena Simona and Nurmi, Jari and Granell, Carlos and Huerta, Joaquín},
  booktitle={2022 23rd IEEE International Conference on Mobile Data Management (MDM)}, 
  title={Data Cleansing for Indoor Positioning Wi-Fi Fingerprinting Datasets}, 
  year={2022},
  volume={},
  number={},
  pages={349-354},
  doi={10.1109/MDM55031.2022.00079}}
```


### Built With

This algorithm has been developed using:
* [Python](https://www.python.org/)


<!-- structure -->
## Getting Started

    .
    ├── datasets                      # WiFi fingerprinting datasets
    ├── cleaned_db                    
    │   ├── csv                       # Cleaned datasets .csv format
    │   ├── mat                       # Cleaned datasets .mat format
    ├── positioning
    │   ├── position.py               # KNN (Classifier and regressor)
    ├── preprocessing
    │   ├── data_cleaning.py          # Data cleaning
    │   ├── data_processing.py        # Normalization, Standardization, ...
    │   ├── data_representation.py    # Positive, Powerd, etc.
    ├── results                       # Positioning results by dataset
    ├── report                        # Generate plots
    │   ├── cdf.py                    # Individual CDF plot
    │   ├── data_size_time.py         # Barplot data size and time
    │   ├── kde_cdf.py                # KDE and CDF plots
    │   ├── removed_fps.py            # Scater plot
    │   ├── PLOTS  
    ├── config.json                   # Configuration file
    ├── main.py                       
    ├── run_cleandb.py                # Run the data cleansing algorithm
    ├── requirements.txt              # Python libraries - requirements
    └── README.md                     # The most important file :)

## Libraries
* pandas, numpy, seaborn, matplotlib, sklearn, colorama

## Datasets 
The datasets can be downloaded either from authors' repository (see README file in dataset folder) or from the following repository:

      "Joaquín Torres-Sospedra, Darwin Quezada-Gaibor, Germán Mendoza-Silva,
      Jari Nurmi, Yevgeny Koucheryavy, & Joaquín Huerta. (2020). Supplementary
      Materials for 'New Cluster Selection and Fine-grained Search for k-Means
      Clustering and Wi-Fi Fingerprinting' (1.0).
      Zenodo. https://doi.org/10.5281/zenodo.3751042"

## Converting datasets from .mat to .csv
1.- Copy the original datasets (.mat) into **dataset** folder.
2.- Modify the list of datasets in the /miscellaneous/datasets_mat_to_csv.py (line 23) with the dataset or datasets to be converted to csv.
```py
list_datasets = [ 'DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'TUT1', 'TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7','UJI1','UTS1', 'UJIB1', 'UJIB2']
```

3.- Run the /miscellaneous/datasets_mat_to_csv.py.
```sh
  $ python /miscellaneous/datasets_mat_to_csv.py
```
## Usage
General parameters:
  * --config-file : Datasets and model's configuration (see config.json)
  * -- dataset : Dataset or datasets to be tested (i.e., UJI1 or UJI1,UJI2,TUT1)
  * -- algorithm : KNN or CLEAN

* **Data cleansing**
  * Execute the data cleansing algorithm + KNN (To generate the cleansed dataset the "clean_db" has to be changed to ** TRUE ** in the config file.)
```sh
  $ python main.py --config-file config.json --dataset LIB1 --algorithm CLEAN
```

<!-- LICENSE -->
## License

CC By 4.0


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
The authors gratefully acknowledge funding from the European Union’s Horizon 2020 Research and Innovation programme under the Marie Sk\l{}odowska Curie grant agreement No. $813278$, A-WEAR.
