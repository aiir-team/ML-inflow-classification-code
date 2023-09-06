
# Machine Learning for Inflow Classification

## Dataset 


1. Raw dataset: data/input_data/inflow_dataset.xlsx
2. Handicraft work to generate file: data/input_data/inflow.csv
3. Using data/input_data/read_data.py to generate inflow_by_mean.csv and inflow_by_prof.csv
4. Current used dataset: data/input_data/inflow_by_mean.csv
5. Select the best features based on ANOVA-test and Mutual Information in file: data/feature-engineering.ipynb

**Please, contact the first author for dataset**


## Models

```code 
ml_models_tuning.py: 
    + Feature engineering with ANOVA test and Mutual Information 
    + Normal dataset and Smote dataset 
    + Tuning all models 

```

Old developed code:
    + smote_rf_tuning.py
    + smote_svm_tuning.py
    + smote_xgb_tuning.py 


## Metrics
From this library: https://github.com/thieu1995/permetrics

- ["AS", "PS", "RS", "F1S", "F2S", "MCC", "LS"] 
- ["accuracy score", "precision score", "recall score", "f1 score", "f2 score", "Matthews Correlation Coefficient", "lift score"]


## Environment management

1. conda or miniconda

```code 

conda create -n new ml python==3.8.5                
conda activate ml
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge scikit-learn
conda install -c conda-forge matplotlib
pip install xlsxwriter==3.0.3
pip install openpyxl==3.0.9
pip install permetrics==1.3.0
pip install mealpy==2.4.2
```

2. pip

2.1 On Windows

```code 
open terminal: (pip on windows is already installed inside python)
   python -m venv ml
   ml\Scripts\activate.bat
   pip install -r requirements.txt
```

2.2 On linux/ubuntu

```code 
sudo apt-get install pip3
sudo apt-get install python3.8-venv

python3 -m venv ml 
source ml/bin/activate
pip install -r requirements.txt
```


- If it is not working, install each of them in order 

```code 
pip install numpy 
pip install xlsxwriter
pip install openpyxl
pip install pandas
pip install seaborn
pip install scikit-learn
pip install matplotlib
pip install mealpy==2.4.2
pip install permetrics==1.3.0
```


## Notes
1. https://albertdchiu.medium.com/a-step-by-step-example-in-binary-classification-5dac0f1ba2dd
2. https://towardsdatascience.com/binary-classification-example-4190fcfe4a3c
3. https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
4. https://medium.com/deep-learning-with-keras/which-activation-loss-functions-part-a-e16f5ad6d82a
5. https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model-9e13838dd3de
6. https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
7. https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
8. https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79
9. https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
10. https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
11. https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79
12. https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
13. https://stackoverflow.com/questions/2186525/how-to-use-glob-to-find-files-recursively
14. https://stackoverflow.com/questions/14509192/how-to-import-functions-from-other-projects-in-python
15. https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python
16. https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time
17. https://datatofish.com/read_excel/
18. https://pythontic.com/modules/pickle/dumps
19. https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018
20. https://github.com/keras-team/keras/issues/14180
21. https://ai-pool.com/d/how-to-get-the-weights-of-keras-model-
