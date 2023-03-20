##################################################################################################
import pandas as pd 
import numpy as np
import seaborn as sns 
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder  
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score

################################################################################################
def Understanding_data(DataFrame): 
  """This is the Function Written in ml_functions.py File. In this Function 
  I'm Understanding the data like "Shape of Data","Information about Data",and
  Seeing the Top 10 rows to understand""" 
  print("***"*300) 
  print("\033[1m" + Understanding_data.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                         #we are creating a Function to understand the given data--> DataFrame is the parameter 
  row_col=pd.DataFrame({"Number_Of_R&C":["Rows","Columns"],"Values":[DataFrame.shape[0],DataFrame.shape[1]]})                                                                           #'.' indicates that we're accessing data or behaviors for a particular object type
  print(row_col)
  print("***"*300)
  print("The Information of the DataSet is :",DataFrame.info(),"\n")         #'.info()'--> it helps to give information about the data set
  print("***"*300)
  print("***"*300)
  print("The Top 10 Rows in the given Dataset are:")
  head=pd.DataFrame(DataFrame.head(10))
  return head

#################################################################################################
def Unique_Null_values(DataFrame): 
  """This is the Function Written in ml_functions.py File. In this Function
  I want to Check for How many "Unique values" and "Null Values" are there in 
  the given data set"""
  print("***"*300) 
  print("\033[1m" + Unique_Null_values.__doc__ + "\033[0m")    # print the docstring
  print("***"*300) 
  dfs=[]                                                           # in this function we are checkinng for Unique,Null  values in columns
  print("The  Total number of Unique values and Null Values  in all columns:\n")
  print("**"*50)
  for col in DataFrame.columns:
    l1=pd.DataFrame({"Column_Name":[col],"Unique_Values":[DataFrame[col].nunique()],"Null_Values":[DataFrame[col].isnull().sum()]})
    dfs.append(l1)
  result=pd.DataFrame(pd.concat(dfs,ignore_index=True))
  return result

#################################################################################################
def Delete_Cardinality(DataFrame,columns=None): 
  """This is the Function Written in ml_functions.py File.
  In this Function I'm "Deleting the High Cardinality Attributes" from the DataFrame""" 
  print("***"*300) 
  print("\033[1m" + Delete_Cardinality.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                      # Cardinality --> having high uniquness i.e., more unique values in column
  print("The DataFrame Befoe Deleting High Cardinality Columns:\n",DataFrame.columns)
  print("***"*200)
  print("***"*200)
  DataFrame.drop(columns=columns,axis=1,inplace=True)                                      #"drop()"--> used to delete the given specified columns in dataset
  print("The DataFrame After Removing High Cardinality Columns:\n",DataFrame.columns)
  print("***"*200)
  print("***"*200)
  return DataFrame

###################################################################################################
def Convert_dtype(DataFrame,date_col,cat_col): 
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Converting the DataTypes into Appropriate 
  formats"""
  print("***"*300) 
  print("\033[1m" + Convert_dtype.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                 
  print("The datatypes before converting :\n",DataFrame.dtypes,'\n')   
  print("**"*50)   
  DataFrame[date_col]=DataFrame[date_col].astype("datetime64")                 #"astype()"--> used to convert the datatype 
  DataFrame[cat_col]=DataFrame[cat_col].astype("category") 
  
  print("The datatypes after converting :\n",DataFrame.dtypes,'\n')
  return DataFrame

###################################################################################################
def count_plot(DataFrame, col1, col2):
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Visualizing Count Plots for any Insights 
  from the Data"""
  fig = px.histogram(DataFrame, x=col1, color=col2, barmode="group")
  fig.update_layout(
      xaxis_title=col1,
      yaxis_title="Count",
      title="Count Plot between {} and {}".format(col1, col2)
  )
  return fig

####################################################################################################
def Separate_x_y_df(DataFrame,Target):   
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Seperating the Independent Attributes and Dependent Attribute 
  which is Required for Further Preprocessing"""                  # in this function we are seperating Indepenndent and dependent variables into two dataframes
  print("***"*300) 
  print("\033[1m" + Separate_x_y_df.__doc__ + "\033[0m")    # print the docstring
  print("***"*300) 
  Y=DataFrame[Target]                                 # taking target variable in to  'Y' Dataframe
  X=DataFrame.drop(columns=Target,axis=1)             # taking all the independent variables into 'X' dataframe excluding target column
  print("The Independent Varaible DataFrame Shape is :\n",X.shape,'\n')
  print("***"*200)
  print("The Dependent Varaible DataFrame Shape is :\n",Y.shape,'\n')
  return X,Y

####################################################################################################
def Split_train_test(X,Y):        
  """This is the Function Written in ml_functions.py File.
  In this Function we are Splitting The Training Data into Train and Validation 
  Data into the Ratio of 70:30""" 
  print("***"*300) 
  print("\033[1m" + Split_train_test.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                                                   # splitting given training data into train and test randomly
  X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=1234)      #train_test_split is the function used for splitting DataFrame into train and test in 70:30 ratio
  print("The shape of X_train is :",X_train.shape,'\n')                                              # stratify=y means we are equalizing/balancing the values in Traget columns  
  print("The shape of Y_train is :",Y_train.shape,'\n') 
  print("The shape of X_test is :",X_test.shape,'\n')
  print("The shape of Y_test is :",Y_test.shape,'\n')
  return X_train,X_test,Y_train,Y_test

####################################################################################################
def Num_Cat_df(DataFrame): 
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Seperating X_train,X_test Data Frames
   into  Numerical and Categorical for Standarizing and Dummification 
   respectively."""    
  print("***"*300) 
  print("\033[1m" + Num_Cat_df.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                      # we are dividng X_train and X_test Data into Categorical and Numerical dataframes 
  num_df = DataFrame.select_dtypes(include=['int','float'])          # here wee are selecting the columns which are in int and float datatyoe then wee store in DataFrame
  cat_df = DataFrame.select_dtypes(include=['category'])             # herre we are storing the columns which are categorical  then store in DataFrame
  print("The Numerical Variables DataFrame Shape is :\n",num_df.shape,'\n')
  print("***"*100)
  print("The Categorical Variables DataFrame Shape is :\n",cat_df.shape,'\n')
  return num_df, cat_df

#####################################################################################################
def One_Hot_Encoding(df,cols):   
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Dummifing the Categorical Variables
  By Using One Hot Encoding by taking Parameter "handle_unknown=ignore"
  which helps in test data"""    
  print("***"*300) 
  print("\033[1m" + One_Hot_Encoding.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                    
  enc = OneHotEncoder(handle_unknown='ignore',drop='first',sparse_output=False)                       # handle_unknown='ignore' means if any new class comes after fitting the data i.e., in test dataset it will just ignore and perform dummifiaction 
  enc = enc.fit(df[cols])                                                         # fitting onehotencode to DataFrame categorical columns
  enc_df=pd.DataFrame(enc.transform(df))                                # transforming onehotencoder i.e., perform dummification on categorical columns
  enc_df.columns = enc.get_feature_names_out(input_features=df.columns)           # get_feature_names_out --> is used to give the names to all columns as like in Previous DataFrame
  print("The Shape of Encoded Data Frame is :",enc_df.shape)
  return enc_df,enc

#####################################################################################################
def Standard_Scaler(X_train_num,X_test_num): 
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Standardizing the Numerical Data By Using 
  Standard_Scaler.""" 
  print("***"*300) 
  print("\033[1m" + Standard_Scaler.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                                   # in this function we are doing Standardization for Scaling the data and convert columns into unit less on numerical columns
  scaler = StandardScaler()                                                             # StandardScaler() is the method using for Standardizing data i.e., calculating Z-score
  scaler=scaler.fit(X_train_num)                                                        # in fit() it calculates overall mean , SD values 
  X_train_num=pd.DataFrame(scaler.transform(X_train_num),columns=X_train_num.columns)   # transform(data) method is used to perform scaling using mean and std dev calculated using the . fit() method.
  X_test_num=pd.DataFrame(scaler.transform(X_test_num),columns=X_test_num.columns)
  print("After Standardizing X_train_num  Shape is:\n",X_train_num.shape,'\n')
  print("**"*50)
  print("After Standardizing X_test_num Shape is:\n",X_test_num.shape,'\n')
  return X_train_num,X_test_num,scaler

######################################################################################################
def Combine_Num_Cat_df(num_df, cat_df):  
  """Combining Numeric And categorical DataFrames After Performing 
  All the preprocessing steps after combining these DataFrame we can
  Directly pass this to models for Prediction"""
  print("***"*300) 
  print("\033[1m" + Combine_Num_Cat_df.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                         # in this method we are combining bboth numerical  and categorical dataframes into one 
  df= pd.concat([num_df,cat_df],axis=1,join='inner')               # for combining we are using pd.concat()--> with hyperparameter join='inner' i.e., combine two dfs which have same records
  print("After combining the shape of df is :\n",df.shape,'\n')
  return df

#######################################################################################################
def ErrorMetrics(actual,pred): 
  """This is the Function Written in ml_functions.py File.
  In this Function I'm Finding the Error Metrics for the 
  model we build""" 
  print("***"*300) 
  print("\033[1m" + ErrorMetrics.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)                                       # error metrics are used to evaluate  our model predictions by giving accuracy,precision,recall
  print("The Error Metrics are :\n","**"*50)
  print("Confusion Matrix \n", confusion_matrix(actual, pred))
  print("Accurcay : ", accuracy_score(actual, pred))
  print("Recall   : ", recall_score(actual, pred))
  print("Precision: ", precision_score(actual, pred)) 
  print("F1 Score:",f1_score(actual,pred))
  print("**"*50)

########################################################################################################
import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
###########################################################################################################
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
def Learning_Curve(model,X,y,model_name):
  """This is the Function Written in ml_functions.py File.
  In this I'm Plotting the Learning Curves For the model."""
  print("***"*300) 
  print("\033[1m" + Learning_Curve.__doc__ + "\033[0m")    # print the docstring
  print("***"*300)
  train_sizes = np.linspace(0.1, 1.0, 10)

  # Compute the learning curve
  train_sizes, train_scores, test_scores = learning_curve(model, X, y, train_sizes=train_sizes, cv=5, scoring='f1')

  # Compute the mean and standard deviation of the scores
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  # Plot the learning curve
  plt.figure()
  plt.title('Learning Curve :{}'.format(model_name))
  plt.xlabel('Number of Training Samples')
  plt.ylabel('F1 Score')
  plt.ylim([0.6, 1.05])
  plt.grid()

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                  train_scores_mean + train_scores_std, alpha=0.1,
                  color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
          label="Training F1 Score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
          label="Cross-validation F1 Score")

  plt.legend(loc="best")
  plt.show()
#########################################################################################################################################