# 🏷️ Retail Price Prediction Machine Learning Engine

An end-to-end machine learning web application designed to estimate optimal e-commerce product retail prices based on competitor catalog patterns, text features, and brand categorizations. 

🔗 **Live App Demo:** [View Live Streamlit App](https://retail-price-predictor-zxyq3gnry5yrcux7wu929w.streamlit.app)

---

## 🚀 Project Overview
In modern e-commerce, setting competitive and profitable retail prices is a critical business driver. Manually auditing competitor catalogs is inefficient and prone to error. This project addresses the challenge by training a machine learning regression pipeline on product catalog patterns, allowing users to instantly forecast prices by analyzing product metadata, brand positioning, and category context.

---

## ✨ Key Features
* **Interactive Web Interface:** Built using Streamlit to provide a clean, user-friendly UI for real-time text input and categorical selection.
* **Scikit-Learn Pipeline Architecture:** Combines text processing and numerical feature scaling seamlessly into a single serialized object.
* **Robust Model Serialization:** Uses joblib to export and load the trained model instantly without retraining overhead.
* **Cloud-Native Deployment:** Deployed and hosted live on Streamlit Community Cloud with pinned Python and library dependencies for seamless execution.

---

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Data Processing & Modeling:** Pandas, Scikit-Learn
* **Model Persistence:** Joblib
* **Web Framework:** Streamlit
* **Version Control & Hosting:** GitHub, Streamlit Cloud

---

## 📂 Repository Structure
* `app.py` - Streamlit frontend application script
* `retail_price_model.joblib` - Serialized machine learning pipeline
* `requirements.txt` - Project environment dependencies

---

## ⚙️ How It Works (Under the Hood)

The prediction workflow follows a structured data science pipeline:

1. **User Input Collection:** 
   * The user enters a descriptive Product Title (e.g., "Classic Fleece Hoodie"), selects a Store Brand from a dropdown, and specifies the Product Category.
2. **Feature Engineering & Transformation:**
   * The application captures the raw text and automatically computes underlying linguistic metadata (such as character length and total word counts).
   * Categorical features (Store Brand and Category) are encoded to align with the training schema.
3. **Pipeline Inference:**
   * The structured input DataFrame is passed directly into the pre-trained joblib pipeline (`retail_price_model.joblib`).
   * The pipeline handles automated vectorization and feature scaling without data leakage.
4. **Price Estimation:**
   * The underlying regression model computes the predicted pricing pattern based on learned competitor structures and displays the estimated retail price instantly on the UI.

---

## 👤 Author
**Ryan**  
*Aspiring Data Science & AI Professional*
