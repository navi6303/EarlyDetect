
![logo](https://github.com/user-attachments/assets/cbb5a1fe-c998-4707-9974-cd8f7371aa1a)  
# **EarlyDetect**
EarlyDetect is a web application designed for early breast cancer detection. It leverages machine learning models trained on the Wisconsin Breast Cancer dataset to predict whether a case is benign or malignant based on fine needle aspirate (FNA) images of breast masses. The app provides fast, accurate predictions and offers a user-friendly interface for healthcare professionals to make informed decisions, improving patient outcomes and reducing anxiety with timely diagnoses.

### PROBLEM STATEMENT
Breast cancer remains a critical public health issue worldwide, with early and accurate diagnosis being essential for successful treatment and improved patient outcomes. Traditional diagnostic methods, while effective, often involve time-consuming processes and may be subject to human error.

Machine learning provides an innovative approach to enhancing breast cancer diagnosis. By leveraging medical data, such as tumor characteristics, a machine learning model can be trained to accurately predict whether a tumor is benign or malignant. This technology has the potential to facilitate faster and more precise diagnoses, enabling timely medical interventions and ultimately saving lives.

A successful machine learning model could have significant business and healthcare impacts, including:

- *Improved patient outcomes*: Early detection and timely treatment can significantly boost survival rates.
- *Reduced healthcare costs*: Efficient and accurate diagnoses can streamline resource allocation and reduce unnecessary procedures.
- *Enhanced patient experience*: Quicker and more reliable diagnoses can alleviate patient anxiety and improve overall satisfaction.
- *Potential for new medical insights*: Analyzing model predictions could lead to a deeper understanding of breast cancer, potentially uncovering novel patterns and risk factors.

### OBJECTIVE
Developing a reliable machine learning model for breast cancer diagnosis can greatly enhance healthcare providers' diagnostic capabilities, leading to better patient care and advancing the field of medical research.

### ABOUT THE DATASET 
The dataset features are derived from digitized images of fine needle aspirates (FNA) of breast masses. These features capture various characteristics of the cell nuclei present in the images. In the 3-dimensional space, the dataset's features are described by the study:

K. P. Bennett and O. L. Mangasarian, "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets," Optimization Methods and Software, vol. 1, 1992, pp. 23-34.

    1. Attribute Information
    2. ID number

    Diagnosis (M = malignant, B = benign)
    3-32. Ten real-valued features computed for each cell nucleus:

    3. radius (mean of distances from center to points on the perimeter)
    4. texture (standard deviation of gray-scale values)
    5. perimeter
    6. area
    7. smoothness (local variation in radius lengths)
    8. compactness (perimeter² / area - 1.0)
    9. concavity (severity of concave portions of the contour)
    10. concave points (number of concave portions of the contour
    11. symmetry
    12. fractal dimension ("coastline approximation" - 1)
The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, and field 23 is Worst Radius.

* Missing attribute values: None
* Class distribution: 357 benign, 212 malignant


## Tech Stack

* **Python 3.11+**: Programming language used for model development and app implementation.

* **Streamlit**: Framework for building and deploying the web application.

* **Pandas**: Library for data manipulation and analysis.

* **scikit-learn**: Library for implementing machine learning models.


## Installation

To run the EarlyDetect web app locally, follow these steps:

```bash
  git clone https://github.com/navi6303/EarlyDetect.git
```

Go to the project directory

```bash
  cd earlydetect
```

Install required packages

```bash
  pip install -r requirements.txt
```

Run the Application

```bash
  streamlit run app.py
```
    
## Deployment

The EarlyDetect web app is also deployed and can be accessed at: [EarlyDetect](https://earlydetect.streamlit.app/)

## Model Performance

| Model | Score     | 
| :-------- | :------- | 
| Logistic Regression | 98.25% |
| Support Vector Classifier | 97.37% |
| Random Forest Classifier | 95.61% | 
| Gradient Boosting Classifier | 95.61% |
| XgBoost | 95.61% |
| K-Nearest Neighbours | 94.74% | 
| Decision Tree Classifier | 92.98% |

**Logistic Regression Results**

The results for Logistic Regression are as follows:

- Precision: 0.99 (benign), 0.98 (malignant)
- Recall: 0.99 (benign), 0.98 (malignant)
- F1-Score: 0.99 (benign), 0.98 (malignant)
- Accuracy: 0.98
- Macro Avg: Precision 0.98, Recall 0.98, F1-Score 0.98
- Weighted Avg: Precision 0.98, Recall 0.98, F1-Score 0.98



## Conclusion

Logistic Regression performed exceptionally well, achieving a score of 98.25%. This high level of accuracy demonstrates the model's effectiveness in capturing the underlying patterns in the data. The results highlight Logistic Regression's suitability for this task, providing robust and reliable predictions. Future work may involve fine-tuning the model further or exploring other algorithms to potentially enhance performance even more.

## Usage

    1. Upload FNA Image Features: Input the relevant features from the FNA images through the sidebar.
    2. Get Predictions: The model will predict whether the case is benign or malignant based on the provided features.
    3. Review Results: View the prediction results and use them to make informed decisions.

## Screenshots
![Screenshot 2024-08-26 034835](https://github.com/user-attachments/assets/d74bebff-8cbd-48b5-9fe3-ec6047b3aab8)

![Screenshot 2024-08-26 034857](https://github.com/user-attachments/assets/af56d056-b3fb-4392-a544-622fe6ce4ac3)



## Acknowledgements

* Wisconsin Breast Cancer Dataset: For providing the data used in training the models.
* Streamlit: For the easy-to-use framework for building web applications.

