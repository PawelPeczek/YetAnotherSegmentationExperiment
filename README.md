# YetAnotherSegmentationExperiment

## Repository set-up

To set up the project:
* Create a conda environment (or venv if you like)
    ```bash
    conda create -n YetAnotherSegmentationExperiment python=3.7
    ```
* Source the env
    ```bash
    conda activate YetAnotherSegmentationExperiment
    ```
* Install requirements
    ```bash
    (YetAnotherSegmentationExperiment) project_root$ pip install -r requirements.txt
    ```
    

__In case of issues with packages visibility in jupyter__
```bash
(YetAnotherSegmentationExperiment) python -m ipykernel install --name "YetAnotherSegmentationExperiment" --user
```

## Data pre-processing

To fetch data:

```bash
python -m src.fetch_data
```

To generate masks:

```bash
python -m src.convert_data
```

## Content navigation
Features provided in this code-base:
* On-line data augmentation - [example](notebooks/AugmentationExperiments.ipynb)
* Dataset splits generation - [example](notebooks/FoldsGenerationDemo.ipynb)
* Automated training of various semantic segmentation models with progress tracing - [example](notebooks/TrainingWrapper.ipynb)
* Automated validation of model performance with detailed results tracing - [example](notebooks/TrainingResultsAnalysis.ipynb)
* Training history plotting - [example](notebooks/TrainingHistoryVisualisation.ipynb)
* LaTeX generator for evaluation results tables - [example](notebooks/LatexTablesGenerator.ipynb)

We prepared a [report](report.pdf) formed in a way as research papers are prepared. 
This report, however, is not intended to be published as the experiment was 
intended to check negative results.


## Citations
```
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```