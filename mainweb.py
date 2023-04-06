import pickle
import streamlit as st 
import numpy as np
import pandas as pd
from PIL import Image
import json
import requests
from streamlit_lottie import st_lottie
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt

def Home():
    st.title("AUTO INSURANCE CLAIM FRAUD DETECTION") 
    st.markdown("<h3 style='color: #FFA31E;'>Welcome to Auto Insurance Claim Fraud Detection Website....! </h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:#FFA31E;'> First  Enter Your Name and Go for Next Steps :)</h3>",unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFA31E;'>You Can Select Any Option From Left Drop Down :) </h3>", unsafe_allow_html=True)
    url = requests.get("https://assets10.lottiefiles.com/packages/lf20_2yyeslc6.json")
    # Creating a blank dictionary to store JSON file,
    # as their structure is similar to Python Dictionary
    url_json = dict()
    
    if url.status_code == 200:
        url_json = url.json()
    else:
        print("Error in the URL")
    
    st_lottie(url_json)

# Template 1: Problem statement and data description
def Problem_Description():
    st.title("AUTO INSURANCE CLAIM FRAUD DETECTION")  
    st.markdown("<h1 style='color: #FFA31E;'>Problem Statement</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00EA87;'>Predicting the “Fraud in auto insurance claims”  &  Pattern extraction</h3>", unsafe_allow_html=True)
    st.write("Our Client is an Insurance Company , which have a Business Problem of Fraud Claims that leads to excessive Leakages So that Our Client wants to Find the Fraudlents ones before even processing the claims to allocate costs appropriately. There are Two main Reasons for the Claims\n")
    st.write("1. Motor Insurance and")
    st.write("2. Health Insurance")
    st.write("These are the segments that have seen a spurt in fraud. Frauds can be classified into following Categories")
    st.subheader("Sources : ")
    st.write("Sources Can be Policy Holders ,Intermediary (i.e., Who are refering the policy to Customers) and Internal (i.e., Employees who working in that Domain).This Problem is more Critical from Internal control framework point of view.")
    st.subheader(" Nature point of view :")
    st.write("Nature Point Of View can refers to application, Inflation , Identity, Fabrication(i.e., Claiming more amount than what they spent), staged/contrived/induced accidents (i.e., Doing Accidents Intentionally To Claim money) etc.")
    st.write("This Kind of Fraud Claims affects the Lives of Innocent people as well as the Insurance industry not only this, Society get Lose their Interest On Insurance Companies which leads to the destruction of Insurance Companies. So, by Detecting the Fraud Claims we can helpful to Regular Customers and Intelligence Team. Predicting at the time of processing claims will reduce costs and minimize losses.")
    st.markdown("<h3 style='color: #FFA31E;'>Data Description</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    # Add the text in the first column
    with col1:
      
        st.markdown("<span style='color:#00EA87'><b>CustomerID</b></span> : Customer ID", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>AmountOfInjuryClaim</b></span> : Claim for injury", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>AmountOfPropertyClaim</b></span> : claim for property damage", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>AmountOfVehicleDamage</b></span> : claim for vehicle damage", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>AmountOfTotalClaim</b></span> : Total claim amount", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>CapitalGains</b></span> : Capital gains(Financial Status)", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>CapitalLoss</b></span> : capital loss(Financial Status)", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>CustomerLoyaltyPeriod</b></span> : Duration of customer relationship", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>PolicyAnnualPremium</b></span> : Annual Premium", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Policy_Deductible</b></span> : Deductible amount", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>UmbrellaLimit</b></span> : Umbrella Limit amount", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>TypeOfIncident</b></span> : Type of incident ", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>TypeOfCollission</b></span> : Type of Collision", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>SeverityOfIncident</b></span> : Collision severity", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>AuthoritiesContacted</b></span> : Which authorities are contacted", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>NumberOfVehicles</b></span> : Number of vehicles involved", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>PropertyDamage</b></span> : If property damage is there", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>BodilyInjuries</b></span> : Number of bodily injuries", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Witnesses</b></span> : Number of witnesses ", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>PoliceReport</b></span> : If police report available", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Insured Gender</b></span> : Gender", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>InsuredEducationLevel</b></span> : Education", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>InsuredOccupation</b></span> : Occupation", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>InsuredHobbies</b></span> : Hobbies", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Policy_CombinedSingleLimit</b></span> : Split Limit and Combined Single Limit", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>InsuredRelationship</b></span> : Realtionship", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Vehicle Make</b></span> : Company Of the Vehicle", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>Vehicle Model</b></span> : Vehicle Model", unsafe_allow_html=True)
        st.markdown("<span style='color:#00EA87'><b>VehicleYOM</b></span> : Vehicle Year Of Make", unsafe_allow_html=True)


    # Add the image in the second column
    with col2:
        image = Image.open("tomandjerry.png")
        st.image(image,width=700)



# Template 2: Data visualization using Pandas profiling
def pandas_profiling():
    # Read the contents of the CSS file
    with open('style.css') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    st.title("AUTO INSURANCE CLAIM FRAUD DETECTION")  
    st.markdown("<h3 style='color: #FFA31E;'>Data Visualization using Pandas Profiling :) </h3>", unsafe_allow_html=True)

    df = pd.read_csv("Train_df.csv")
    profile = ProfileReport(df, title="Pandas Profiling Report")
    st_profile_report(profile)


def predict_auto_insurance_claim_fraud(TypeOfIncident,TypeOfCollission, SeverityOfIncident,AuthoritiesContacted, NumberOfVehicles,PropertyDamage,
            BodilyInjuries, Witnesses, PoliceReport, AmountOfTotalClaim,AmountOfInjuryClaim, AmountOfPropertyClaim, AmountOfVehicleDamage,InsuredAge, InsuredGender,
            InsuredEducationLevel,InsuredOccupation,InsuredHobbies, CapitalGains, CapitalLoss,CustomerLoyaltyPeriod, Policy_CombinedSingleLimit,Policy_Deductible, 
            PolicyAnnualPremium, UmbrellaLimit,InsuredRelationship, VehicleMake, VehicleModel, VehicleYOM):
    # Transform categorical variables using one-hot encoding
            # Load the pickle files containing the transformers

    with open('onehotencoder.pkl', 'rb')as f2:
        onehotencoder = pickle.load(f2)

    with open('std_scaler.pkl', 'rb') as f3:
        standardization = pickle.load(f3)

    # Load the pickle file containing the classifier
    with open('svc_clf.pkl', 'rb') as f4:
        classifier = pickle.load(f4)
    categorical_data = [[TypeOfIncident, TypeOfCollission, SeverityOfIncident,AuthoritiesContacted,NumberOfVehicles, PropertyDamage,BodilyInjuries,Witnesses,
                                    PoliceReport,InsuredGender,InsuredEducationLevel,InsuredOccupation,InsuredHobbies,Policy_CombinedSingleLimit,InsuredRelationship, 
                                    VehicleMake,VehicleModel,VehicleYOM]]
    categorical_data = onehotencoder.transform(categorical_data)

        # Transform numerical variables using standardization
    numerical_data = [[ AmountOfTotalClaim,AmountOfInjuryClaim, AmountOfPropertyClaim, AmountOfVehicleDamage,InsuredAge,CapitalGains, CapitalLoss,
    CustomerLoyaltyPeriod,Policy_Deductible, PolicyAnnualPremium, UmbrellaLimit]]
    numerical_data = standardization.transform(numerical_data)

        # Concatenate the transformed data
    input_data = np.concatenate((categorical_data, numerical_data), axis=1)

        # Predict diabetes using the classifier
    prediction = classifier.predict(input_data)
    return prediction
    
# Template 3: Prediction page
def prediction_page(name):
    import streamlit_theme as stt
    # Read the contents of the CSS file
    with open('style.css') as f:
        css = f.read()

    # Add the CSS to the app's page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.title("AUTO INSURANCE CLAIM FRAUD DETECTION")  
    CustomerID=st.text_input("Customer ID","Cust10008")
    AmountOfInjuryClaim = st.number_input("Amount Of Injury Claim",value=6835, help="Enter the claim amount in Rupees")
    AmountOfPropertyClaim = st.number_input("Amount Of Property Claim",value=8059, help="Enter the claim amount in Rupees")
    AmountOfVehicleDamage = st.number_input("Amount Of Vehicle Damage",value=53460, help="Enter the damage amount in Rupees")
    AmountOfTotalClaim = st.number_input("Amount Of Total Claim",value=68354, help="Enter the amount which is Total of all above claims ")
    InsuredAge = st.number_input("Insured Age",value=27, help="Enter the age of the insured in years")
    CapitalGains = st.number_input("Capital Gains",value=56400, help="Enter the capital gains in Rupees")
    CapitalLoss = st.number_input("Capital Loss",value=-57000, help="Enter the capital loss in Rupees")
    CustomerLoyaltyPeriod = st.number_input("Customer Loyalty Period",value=84,help="Enter the number of days the customer has been with the company")
    Policy_Deductible = st.number_input("Policy Deductible", value=2000,help="Enter the policy deductible in Rupees")
    PolicyAnnualPremium = st.number_input("Policy Annual Premium",value=1006, help="Enter the policy annual premium in Rupees")
    UmbrellaLimit = st.number_input("Umbrella Limit",value=0, help="Enter the umbrella limit in Rupees")
    
    # Dropdown options for categorical variables
    TypeOfIncident= st.selectbox("Type Of Incident", ["Multi-vehicle Collision","Single Vehicle Collision","Parked Car","Vehicle Theft"],index=0)
    TypeOfCollission= st.selectbox("Type Of Collission", ["Rear Collision","Side Collision","Front Collision" ],index=2)
    SeverityOfIncident= st.selectbox("Severity Of Incident", ["Minor Damage","Total Loss","Major Damage","Trivial Damage"],index=0)
    AuthoritiesContacted= st.selectbox("Authorities Contacted", ["Police","Fire","Ambulance","Other",'None'],index=2)
    NumberOfVehicles= st.selectbox("Number Of Vehicles", ['1','2','3','4'],index=2)
    PropertyDamage= st.selectbox("Property Damage", ['NO','YES'],index=1)
    BodilyInjuries= st.selectbox("Bodily Injuries", ['0','1','2'],index=0)
    Witnesses= st.selectbox("Witnesses", ['0.0','1.0','2.0','3.0'],index=0)
    PoliceReport= st.selectbox("Police Report", ['NO','YES'],index=0)
    InsuredGender= st.selectbox("Insured Gender", ['MALE','FEMALE'],index=1)
    InsuredEducationLevel= st.selectbox("Insured Education Level", ['JD','High School','MD','Masters','Associate','PhD','College'],index=1)
    InsuredOccupation= st.selectbox("Insured Occupation", ["machine-op-inspct","prof-specialty","tech-support","priv-house-serv","exec-managerial","sales","craft-repair","transport-moving",
                                                            "armed-forces","othe-service","adm-clerical","protective-serv","farming-fishing","handlers-cleaners"],index=10)
    InsuredHobbies= st.selectbox("Insured Hobbies", ["bungie-jumping","paintball","camping","kayaking","exercise","reading","movies","yachting","hiking","base-jumping","golf","video-games",
                                                        "board-games","skydiving","polo","cross-fit","sleeping","dancing","chess","basketball"],index=9)
    Policy_CombinedSingleLimit = st.selectbox("Policy Combined Single Limit", ["250/500","100/300","500/1000","250/300","100/500","250/1000","500/500","500/300","100/1000"],index=2)
    InsuredRelationship= st.selectbox("Insured Relationship", ["own-child",'not-in-family',"other-relative","husband","wife","unmarried"],index=0)
    VehicleMake= st.selectbox("Vehicle Make", ["Saab","Suburu","Nissan","Dodge","Chevrolet","Ford" ,"Accura","BMW","Toyota","Volkswagen" ,"Audi","Jeep","Mercedes","Honda"],index=9)
    VehicleModel= st.selectbox("Vehicle Model", ["RAM","Wrangler","A3" ,"MDX","Jetta","Neon","Pathfinder","Passat","Legacy",'92x' ,"Malibu","95","A5" ,"F150" ,"Forrestor","Camry",
                                                    "Tahoe","93","Maxima","Grand Cherokee","Escape","Ultima","E400","X5","TL","Silverado","Fusion","Highlander","Civic","ML350","Impreza","CRV","Corolla",
                                                    "M5","C300" ,"X6","3 Series","RSX" ,"Accord"],index=7)
    VehicleYOM= st.selectbox("Vehicle YOM", ['1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011',
                                                '2012','2013','2014','2015'],index=0)
    result = ""
    if st.button("Predict"):
        prediction = predict_auto_insurance_claim_fraud(TypeOfIncident,TypeOfCollission, SeverityOfIncident,AuthoritiesContacted, NumberOfVehicles,PropertyDamage,
        BodilyInjuries, Witnesses, PoliceReport, AmountOfTotalClaim,AmountOfInjuryClaim, AmountOfPropertyClaim, AmountOfVehicleDamage,InsuredAge, InsuredGender,
        InsuredEducationLevel,InsuredOccupation,InsuredHobbies, CapitalGains, CapitalLoss,CustomerLoyaltyPeriod, Policy_CombinedSingleLimit,Policy_Deductible, 
        PolicyAnnualPremium, UmbrellaLimit,InsuredRelationship, VehicleMake, VehicleModel, VehicleYOM)
        if prediction==1:
            result="Fraud"
        else:
            result="Not A Fraud"
        Result_page(name,CustomerID,result)
         
        
    
# Template 4: Results page
def Result_page(name,CustomerID,result):
    # Read the contents of the CSS file
    with open('style.css') as f:
        css = f.read()

    # Add the CSS to the app's page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st.subheader("Hello,{}...! We are Very Happy for Using Our Service".format(name))

    st.write("Based on the Provided Details, We are Predicting that the Customer with ID {}   is ".format(CustomerID))
    st.subheader(result)
    st.write("We are Giving Our Predictions With ")

    st.subheader("Accuracy  :94%")
    st.subheader("Precision :97%")
    st.subheader("Recall    :83%")
    st.subheader("F1_Score  :89%")
    
def main():
    """
    This function defines the main function to run the web page.
    """
    st.set_page_config(page_title="AUTO INSURANCE CLAIM FRAUD DETECTION APP",page_icon=":guardsman:",layout="wide",initial_sidebar_state="expanded")
    # Create the sidebar menu
    with open('style.css') as f:
        css = f.read()

    # Add the CSS to the app's page
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
    # Define the sidebar menu
    menu = ["Home","Problem Statement and Data Description", "OverView Of Data", "Prediction Page"]

    # Add input fields for name and details in the sidebar
    custom_css = """
        <style>
            .dropdown-menu { 
                background-color: white !important;
            }
        </style>
    """

    with st.sidebar:
        st.subheader("User Information")
        name = st.text_input("User Name")
        
    # Show the appropriate page based on the user's choice
        st.subheader("Select Your Choice")

        choice = st.sidebar.selectbox("Select a page", menu)
        url = requests.get("https://assets8.lottiefiles.com/packages/lf20_goa8injd.json")
        # Creating a blank dictionary to store JSON file,
        # as their structure is similar to Python Dictionary
        url_json = dict()
        
        if url.status_code == 200:
            url_json = url.json()
        else:
            print("Error in the URL")
        
        st_lottie(url_json)
    if choice == "Home":
        Home()
    elif choice == "Problem Statement and Data Description":
        Problem_Description()
    elif choice == "OverView Of Data":
        pandas_profiling()
    elif choice == "Prediction Page":
        prediction_page(name)  



if __name__ == '__main__':
    main()
