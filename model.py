
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import pickle
from sklearn.model_selection import train_test_split

# Load the model from the file
with open('Models/randomForest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)


print("Random Forest Model Loaded")
import pandas as pd
df=pd.read_csv('Credit_card_fraud_detection.csv')
x=df.drop(labels=['default payment next month'],axis=1)
y=df['default payment next month']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=20)
ss=StandardScaler()
s_train=ss.fit_transform(X_train)
s_test=ss.transform(X_test)
print("Model and test data loaded")

# y_pred=rf_model.predict(s_test)
# model_accuracy=accuracy_score(y_test,y_pred)*100

import streamlit as st

st.title("Credit Card Fraud Detection")

def homePage():
   
    st.write(df.head())
c_len=len(df.columns)
train_len=len(s_train)
test_len=len(s_test)   

def render_option_1():
    st.title("Random Forest model")
    st.write("This is custom content for VGG.")

    data = {
    'Data' : ['No of Columns',"Rows"],
    'Train': [c_len, train_len],
    'Test': [c_len, test_len],
    # 'Accuracy':[model_accuracy]
        }
    df = pd.DataFrame(data)

    st.table(df)

    

    if st.button('Test Accuracy'):
        st.write("Testing Started")
        predict=rf_model.predict(s_test)
        st.write(f"model accuracy{accuracy_score(y_test,predict)}")
# Create input fields for each feature
l=[]
for i in range(len(df.columns)):
    input1=st.number_input(f"Enter Injput value for {df.columns[i]}:")
    l.append(input1)
# Add more input fields for other features as needed
dictin={}
j=0
for i in df.columns:
    dictin[i]=l[j]
    j+=1


# Create a button to trigger prediction
if st.button("Predict"):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame(dictin)

    # Predict the label for the input data
    prediction = rf_model.predict(input_data)

    # Display the prediction
    st.write("Prediction:", prediction)


    




           



    
# # def render_option_2():
#     st.title("VGG16 Model")
#     st.write("This is custom content for VGG16.")
#     # You can add more content specific to Option 2 here

#     data = {
#     'Data' : ['No of images',"per of data","Accuracy"],
#     'Train': [6000, 60, 99],
#     'Test': [1500, 25, 95.4],
#     'Validation': [2500,25, 98]
#         }
#     df = pd.DataFrame(data)

#     st.table(df)

#     if st.button('Test Accuracy'):
#         st.write("Testing Started")
#         predict=vgg_16.predict(test_data)
#         predict_=np.argmax(predict,axis=1)
#         tr=test_data.classes
#         st.write(f"model accuracy{accuracy_score(tr,predict_)}")

#     uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image',  width=224,)
#         if st.button('Perform Inference'):
#             if 'model' not in locals():
#                 st.error('Please load the model first!')
           


    

# def render_option_3():
#     st.title("SVM model")
#     st.write("This is custom content for SVM.")
#     # You can add more content specific to Option 3 here

#     data = {
#     'Data' : ['No of images',"per of data","Accuracy"],
#     'Train': [6000, 60, 99],
#     'Test': [1500, 25, 95.4],
#     'Validation': [2500,25, 98]
#         }
#     df = pd.DataFrame(data)

#     st.table(df)

#     if st.button('Test Accuracy'):
        
#         st.success('Model loaded successfully!')

#     uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image',  width=224)
#         if st.button('Perform Inference'):
#             if 'model' not in locals():
#                 st.error('Please load the model first!')
           





def side_bar():
    st.sidebar.title("Sidebar with Dropdown")
    
    # Add a dropdown in the sidebar
    dropdown_options = ["Select","Random Forest", "VGG16", "SVM"]
    selected_option = st.sidebar.selectbox("Select an option", dropdown_options)
    
    # Render content based on the selected option
    if selected_option == dropdown_options[1]:
        render_option_1()
    elif selected_option == dropdown_options[2]:
        # render_option_2()
        pass
    elif selected_option == dropdown_options[3]:
        # render_option_3()
        pass
    else:
        homePage()



side_bar()