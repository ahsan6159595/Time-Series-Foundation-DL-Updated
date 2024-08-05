Overview

This project aims to detect anomalies in time series data using the Moment model. The process involves data preprocessing, model training, fine-tuning, and evaluation using three different datasets.

Contents

anomaly_detection.ipynb: Jupyter notebook for anomaly detection.
anomaly_detection_dataset.py: Script for loading and processing datasets.
datasets/: Directory containing the datasets.
models/: Directory containing pre-trained models.

Prerequisites
Momentfm
Python 3.11
Jupyter Notebook
Required Python libraries (install using pip install -r requirements.txt)

1. Installation
Clone the Repository:
git clone https://github.com/yourusername/anomaly-detection.git
or run locally:
Dataset File: "anomaly_detection_dataset.py"
Model File: "anomaly_detection.ipynb" 

2.Install Dependencies:
pip install -r requirements.txt
pip install momentfm

Datasets
Place your datasets in the datasets/ directory. Ensure the filenames match the following:

i)	133_UCR_Anomaly_InternalBleeding14_2800_5607_5634.txt
ii)	155_UCR_Anomaly_PowerDemand4_18000_24005_24077.txt
iii)	113_UCR_Anomaly_CIMIS44AirTemperature1_4000_5391_5392.txt
   	Running the Code

(a)Open the Jupyter Notebook:

jupyter notebook anomaly_detection.ipynb

(b)Load the Dataset:

In the notebook, specify the path to the dataset you want to use. For example:
dataset_path = '133_UCR_Anomaly_InternalBleeding14_2800_5607_5634.txt'

3. Preprocess the Data:

Run the cells that preprocess the data. This will involve normalizing and splitting the data into training and testing sets.

4. Train the Model:

Execute the cells that train the Moment model on the training dataset. This includes both initial training and fine-tuning steps.

5. Evaluate the Model:

Run the cells to evaluate the model's performance on the test dataset. Metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) will be calculated.

6. Plot the Results:

Generate plots to visualize the observed, predicted, and anomaly scores. Ensure the indices for the anomaly range are set correctly.

python code:
anomaly_start = 2800
anomaly_end = 5607
start = anomaly_start - 512
end = anomaly_end + 512

plt.plot(trues[start:end], label="Observed", c='darkblue')
plt.plot(preds[start:end], label="Predicted", c='red')
plt.plot(anomaly_scores[start:end], label="Anomaly Score", c='black')
plt.legend(fontsize=16)
plt.show()
Repeat for Other Datasets:

Follow the same steps for the other two datasets:
155_UCR_Anomaly_PowerDemand4_18000_24005_24077.txt
113_UCR_Anomaly_CIMIS44AirTemperature1_4000_5391_5392.txt