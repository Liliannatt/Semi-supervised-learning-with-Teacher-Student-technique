# Semi-supervised-learning-with-Teacher-Student-technique

This project explores a semi-supervised learning strategy to reduce the reliance on large amounts of labeled data in Automatic Speech Recognition (ASR) systems. By combining a small labeled dataset with a larger unlabeled dataset, we aim to enhance ASR performance while significantly decreasing the need for extensive manual labeling.

## Methods

Teacher Model Training
- Dataset: Limited labeled dataset from the TIMIT corpus.
- Model: A complex Long Short-Term Memory (LSTM)-based neural network.
- Training: The teacher model is trained to achieve high accuracy on the labeled data.

Soft Label Generation
- Unlabeled Data: A larger corpus of unlabeled speech data.
- Soft Labels: The trained teacher model generates probabilistic outputs (soft labels) for the unlabeled data.
- Purpose: Soft labels provide rich information about the uncertainty and relationships between classes.

Student Model Training
- Model: A simpler LSTM-based neural network.
- Training: The student model is trained on the unlabeled data using the soft labels from the teacher model.
- Objective: To mimic the teacher model's performance while being more efficient.

