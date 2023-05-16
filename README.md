# InterShip Projects
In the course of my internship at Corizo, I completed these two Minor and Major projects.

## stock price prediction project
Process

    Data Collection: Acquired the stock price dataset from Corizo, the source of the dataset during your internship. The dataset encompassed historical stock price data for the desired stock(s) and covered a specific timeframe (e.g., daily data for the past five years).

    Data Preprocessing: Performed essential preprocessing steps on the dataset to prepare it for model training. The preprocessing steps involved were:
        Handling missing values: Identified and dealt with any missing data points in the dataset. Implemented a strategy, such as forward-filling, backward-filling, or interpolation, to address the missing values appropriately.
        Feature engineering: Extracted relevant features from the raw stock price data to enhance the models' predictive power. Features could include moving averages, technical indicators, or other domain-specific factors.
        Data scaling: Applied a scaling technique, such as normalization or standardization, to ensure all features were within a consistent range and to avoid biases towards certain features.

    Dataset Splitting: Split the preprocessed dataset into training and testing sets. Typically, an 80:20 or 70:30 ratio was used, where a portion of the data (e.g., 80%) was allocated for training the models, and the remaining portion was used for testing and evaluating the models' performance.

    Model Training - LSTM: Implemented the LSTM model using a deep learning framework, such as TensorFlow or PyTorch. The steps involved in training the LSTM model were:
        Architecture: Designed an appropriate LSTM architecture for the stock price prediction task, considering factors such as the number of LSTM layers, the number of hidden units per layer, and the activation functions to use.
        Loss function and optimizer: Selected a suitable loss function, such as mean squared error (MSE), and an optimizer, such as Adam or RMSprop, for training the LSTM model.
        Hyperparameter tuning: Experimented with different hyperparameters, such as learning rate, batch size, and number of epochs, to find optimal settings for the LSTM model.
        Training: Fed the preprocessed training data into the LSTM model, performed forward and backward propagation, and updated the model's weights iteratively to minimize the loss function.

    Model Training - XGBoost: Employed the XGBoost algorithm using the XGBoost library or a similar gradient boosting framework. The steps involved in training the XGBoost model were:
        Data preparation: Formatted the preprocessed data into the appropriate input format for XGBoost, such as a pandas DataFrame or DMatrix.
        Hyperparameter tuning: Conducted a hyperparameter search using techniques like grid search or random search to identify the optimal hyperparameter configuration for the XGBoost model. Explored parameters such as learning rate, maximum depth of trees, and the number of estimators.
        Training: Trained the XGBoost model on the preprocessed training data using the selected hyperparameters. Monitored the training progress and assessed the model's performance using evaluation metrics.

    Model Evaluation: Evaluated the performance of both the LSTM and XGBoost models using appropriate evaluation metrics, such as root mean squared error (RMSE), mean absolute error (MAE), or mean percentage error (MPE). Compared the performance of the two models on the test set to assess their relative effectiveness in predicting stock prices.

    Prediction: Applied the trained LSTM and XGBoost models to make predictions on new, unseen stock price data. Analyzed the predictions and assessed their accuracy and reliability using evaluation metrics and visualizations.

Throughout the project, detailed notes were taken on the preprocessing steps, model configurations, and performance evaluations.

## red wine quality analysis project
Process

    Data Collection: Received the red wine quality dataset provided by the internship organization. The dataset contained features related to red wine samples, such as acidity levels, pH, alcohol content, and other chemical properties, along with corresponding quality ratings.

    Data Preprocessing: Performed essential preprocessing steps on the dataset to prepare it for model training. The preprocessing steps included:
        Handling missing values: Identified any missing data points and implemented an appropriate strategy to handle them, such as imputation with mean or median values.
        Feature scaling: Applied feature scaling techniques, such as standardization or normalization, to ensure that all features were on a similar scale and avoid biases towards certain features.
        Feature encoding: Encoded categorical features, if any, into numerical representations suitable for the classification models.

    Dataset Splitting: Split the preprocessed dataset into training and testing sets. Typically, a 70:30 or 80:20 ratio was used, where a portion of the data (e.g., 70%) was allocated for training the models, and the remaining portion was used for testing and evaluating the models' performance.

    Model Training: Utilized various classification models for the red wine quality analysis. The models used included Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Tree, Gaussian Naive Bayes, Random Forest, XGBoost, MLPClassifier, and Artificial Neural Networks (ANN).
        Model Configuration: Set up the appropriate configuration for each classification model, including hyperparameter settings, such as the number of neighbors for KNN or the maximum depth for Decision Trees.
        Model Training: Trained each classification model using the preprocessed training data. Adjusted the models' parameters iteratively using techniques like grid search or random search to optimize their performance.
        Evaluation Metrics: Evaluated the performance of each classification model using relevant evaluation metrics, such as accuracy, precision, recall, and F1-score. Compared the results of the models to assess their relative performance.

    Model Selection - Random Forest: Based on the evaluation results, determined that the Random Forest model yielded the highest accuracy among the tested classification models.

    Hyperparameter Tuning: Fine-tuned the hyperparameters of the Random Forest model to optimize its performance. Conducted a grid search or random search to find the best combination of hyperparameters, considering factors like the number of trees, maximum depth, and feature subset size.

    Model Evaluation: Assessed the performance of the tuned Random Forest model on the preprocessed testing data. Calculated evaluation metrics, such as accuracy, precision, recall, and F1-score, to measure the model's effectiveness in predicting the quality of red wines.

    Prediction: Applied the trained and tuned Random Forest model to make predictions on new, unseen red wine samples. Analyzed the predictions and evaluated the model's accuracy and reliability in predicting the quality ratings.

Throughout the project, comprehensive notes were taken, documenting the preprocessing steps, model configurations, hyperparameter tuning, and evaluation results.
