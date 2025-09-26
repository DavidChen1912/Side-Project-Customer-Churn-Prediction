# Side Project – Customer Churn Prediction  

---

## 📖 Introduction  
Imagine working at a telecom company facing a rising customer churn rate.  
To address this challenge, I designed a **Random Forest classification model** to both quantify the impact of key factors on churn and predict which customers are likely to churn.  

The project pipeline includes:  
1. **Data preprocessing**  
2. **Outlier removal** using the IQR method  
3. **Skewness correction** via Box-Cox transformation  
4. **Class imbalance handling** with SMOTENC  
5. **Feature engineering** to reduce dimensionality  
6. **Hyperparameter tuning** through cross-validation  
7. **Model evaluation** with confusion matrix, Precision, Recall, and F1-score  

Additional project materials are stored in the [`/docs`](./docs) directory. 
Dataset: [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📊 Visualization  
More interactive results can be explored on my [Shiny App](https://jingweinccu.shinyapps.io/telcoproject/), which features:  
1. **Basic EDA**: pie charts, bar plots, density plots, and more  
2. **FAMD Analysis**: a PCA-like dimensionality reduction technique that highlights the most significant principal components (features)  

---

## 🚀 Usage  

Run the following command in your terminal:  

```
python execute.py --data=Churn
```


