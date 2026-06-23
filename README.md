# Real-Time Credit Card Fraud Detection System

Developed a real-time credit card fraud detection system capable of identifying fraudulent transactions with low latency. Built a streaming data pipeline using Apache Kafka for transaction ingestion and Pathway for real-time stream processing. Trained and deployed an XGBoost-based classification model to analyze incoming transaction data and predict fraudulent activity in real time.

The system incorporates feature engineering, data preprocessing, and model evaluation techniques to achieve high detection accuracy while minimizing false positives. Kafka producers simulate live transaction streams, while consumers process and classify transactions as they arrive. The project demonstrates the integration of machine learning with modern stream-processing technologies for scalable fraud detection applications.

### Tech Stack

* Python
* Apache Kafka
* Pathway
* XGBoost
* Pandas
* Scikit-learn

### Key Features

* Real-time transaction ingestion and processing
* Machine learning-based fraud classification
* Low-latency streaming architecture
* Scalable event-driven pipeline using Kafka
* Performance evaluation using industry-standard metrics (Precision, Recall, F1-Score, ROC-AUC)
