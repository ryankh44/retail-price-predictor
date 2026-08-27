import joblib
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title='Retail Price Predictor', page_icon='🏷️', layout='centered'
)

st.title('🏷️ Retail Price Prediction Engine')
st.write(
    'Estimate product retail prices based on competitor catalog patterns using'
    ' machine learning.'
)


# Load trained model
@st.cache_resource
def load_pipeline():
  return joblib.load('retail_price_model.joblib')


try:
  model = load_pipeline()
except Exception as e:
  st.error(
      'Model file standard loading failed. Make sure retail_price_model.joblib'
      ' is in the root directory.'
  )
  st.stop()

# User Input Form
st.subheader('Enter Product Details')

product_name = st.text_input(
    'Product Title',
    value='Classic Fleece Hoodie',
    help='Enter the title or name of the item',
)

col1, col2 = st.columns(2)

with col1:
  store = st.selectbox(
      'Store Brand', options=['Kith', 'Chubbiesshorts', 'Aloyoga', 'Other']
  )

with col2:
  category = st.selectbox(
      'Product Category',
      options=[
          'Tops',
          'Bottoms',
          'Outerwear',
          'Accessories',
          'Footwear',
          'Uncategorized',
      ],
  )

# Process inputs on button click
if st.button('Predict Price', type='primary'):
  if not product_name.strip():
    st.warning('Please enter a product title.')
  else:
    # Compute title length features matching our training pipeline
    title_length = len(product_name)
    word_count = len(product_name.split())

    # Build input DataFrame
    input_data = pd.DataFrame([{
        'Product_Name': product_name,
        'Store': store,
        'Category': category,
        'Title_Length': title_length,
        'Word_Count': word_count,
    }])

    # Predict price
    predicted_price = model.predict(input_data)[0]

    # Display Result
    st.success(f'### Predicted Retail Price: **${predicted_price:.2f}**')

    # Breakdown metrics
    st.markdown('---')
    st.caption('**Feature Breakdown:**')
    st.json({
        'Product Title': product_name,
        'Store': store,
        'Category': category,
        'Character Length': title_length,
        'Word Count': word_count,
    })
