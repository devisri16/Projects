Titanic Classification

The provided code is a Python implementation for predicting the survival outcome of individual passengers in the Titanic dataset using a trained Random Forest Classifier. The steps involved are as follows:

1. Load the Titanic dataset and perform data preprocessing, which includes filling missing 'Age' values with the median age, converting 'Sex' to numerical representation (0 for male, 1 for female), and one-hot encoding the 'Embarked' feature.
2. Select relevant features ('Pclass', 'Age', 'Sex', 'Embarked_Q', 'Embarked_S') for training the Random Forest Classifier.
3. Initialize the Random Forest Classifier and train it on the entire preprocessed dataset.
4. The `predict_survival` function is defined to predict the survival outcome for a specific person. It takes a person's information as input (e.g., 'Pclass', 'Age', 'Sex').
5. Create a DataFrame for prediction using the provided person_info, ensuring 'Sex' is converted to numerical representation, and set default values for 'Embarked_Q' and 'Embarked_S' to 0.
6. Use the trained model to predict the survival outcome for the specific person.
7. The function returns either 'Survived' or 'Not Survived' based on the prediction.
8. Example usage is shown, demonstrating how to call the `predict_survival` function with a dictionary containing 'Pclass', 'Age', and 'Sex' information for a specific person.
9. The code aims to provide quick and individualized survival predictions for passengers without the 'Survived' information.
10. Please note that the accuracy of the predictions depends on the quality of the trained model and the representativeness of the data provided for prediction.
