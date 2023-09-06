
# code_inflow_classification

## Dataset 

1. Raw dataset: data/input_data/inflow_dataset.xlsx
2. Handicraft work to generate file: data/input_data/inflow.csv
3. Using data/input_data/read_data.py to generate inflow_by_mean.csv and inflow_by_prof.csv
4. Current used dataset: data/input_data/inflow_by_mean.csv
5. Select the best features based on ANOVA-test and Mutual Information in file: data/feature-engineering.ipynb


## Models

```code 
1. ml_models_tuning.py: 
    + Feature engineering with ANOVA test and Mutual Information 
    + Normal dataset and Smote dataset 
    + Tuning all models 

2. smote 
    + smote_rf_tuning.py
    + smote_svm_tuning.py
    + smote_xgb_tuning.py 
- Not use feature engineering yet 

```


Now, in this work, we ONLY work with reservoir inflow, which we need ONLY to divide it into 5 classes
So, we need the model to give us the class of the future inflow in time (t+1),

To do that, we need to examine different machine learning model and also different scenarios of the input ((t), (t-1), 
(t-2), ...., (t-n) to make prediction for the inflow class at (t+1)
and it will be very good if we can make Multi-lead ahead prediction, to (t+2) and (t+3)


## Metrics

- ["AS", "PS", "RS", "F1S", "F2S", "MCC", "LS"] 
- ["accuracy score", "precision score", "recall score", "f1 score", "f2 score", "Matthews Correlation Coefficient", "lift score"]


## Cài đặt môi trường 

1. Nếu dùng conda hoặc miniconda thì cài theo thứ tự như sau 

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

2. Nếu dùng pip thì cài bằng lệnh:

2.1 Trên windows

```code 
open terminal: (pip on windows is already installed inside python)
   python -m venv ml
   ml\Scripts\activate.bat
   pip install -r requirements.txt
```

2.2 Trên linux/ubuntu

```code 
sudo apt-get install pip3
sudo apt-get install python3.8-venv

python3 -m venv ml 
source ml/bin/activate
pip install -r requirements.txt
```


- Nếu không được thì cài lần lượt từng thư viện một như sau 

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
https://albertdchiu.medium.com/a-step-by-step-example-in-binary-classification-5dac0f1ba2dd
https://towardsdatascience.com/binary-classification-example-4190fcfe4a3c
https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
https://medium.com/deep-learning-with-keras/which-activation-loss-functions-part-a-e16f5ad6d82a


1. https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model-9e13838dd3de
2. https://www.scikit-yb.org/en/latest/api/classifier/rocauc.html
3. https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
4. https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79

5. https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
6. https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
7. https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79

8. https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
9. https://stackoverflow.com/questions/2186525/how-to-use-glob-to-find-files-recursively
10. https://stackoverflow.com/questions/14509192/how-to-import-functions-from-other-projects-in-python
11. https://stackoverflow.com/questions/51520/how-to-get-an-absolute-file-path-in-python
12. https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time


13. https://datatofish.com/read_excel/

14. https://pythontic.com/modules/pickle/dumps
15. https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018
16. https://github.com/keras-team/keras/issues/14180
17. https://ai-pool.com/d/how-to-get-the-weights-of-keras-model-

```python 
https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list

import json 
x = "[0.7587068025868327, 1000.0, 125.3177189672638, 150, 1.0, 4, 0.1, 10.0]"
solution = json.loads(x)
print(solution)

```