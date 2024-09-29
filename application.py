from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

preprocessor = pickle.load(open('./models/preprocessor.pkl', 'rb'))
final_model = pickle.load(open('./models/final_trained_model.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('index.html')  

@app.route('/landing')  
def landing():
    return render_template('home.html', results=None) 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    results = None 
    if request.method == "POST":
        try:
            Age = request.form.get('Age')
            TypeofContact = request.form.get('TypeofContact')
            CityTier = request.form.get('CityTier')
            DurationOfPitch = request.form.get('DurationOfPitch')
            Occupation = request.form.get('Occupation')
            Gender = request.form.get('Gender')
            NumberOfFollowups = request.form.get('NumberOfFollowups')
            ProductPitched = request.form.get('ProductPitched')
            PreferredPropertyStar = request.form.get('PreferredPropertyStar')
            MaritalStatus = request.form.get('MaritalStatus')
            NumberOfTrips = request.form.get('NumberOfTrips')
            Passport = request.form.get('Passport')
            PitchSatisfactionScore = request.form.get('PitchSatisfactionScore')
            OwnCar = request.form.get('OwnCar')
            Designation = request.form.get('Designation')
            MonthlyIncome = request.form.get('MonthlyIncome')
            TotalVisiting = request.form.get('TotalVisiting')

            # Check for missing values (using empty string instead of None)
            if '' in [Age, TypeofContact, CityTier, DurationOfPitch, Occupation, Gender,
                       NumberOfFollowups, ProductPitched, PreferredPropertyStar, MaritalStatus,
                       NumberOfTrips, Passport, PitchSatisfactionScore, OwnCar, Designation,
                       MonthlyIncome, TotalVisiting]:
                return render_template('home.html', results="Error: Please fill all the fields.")

            # Convert to correct types
            input_data = pd.DataFrame([[float(Age), TypeofContact, int(CityTier), float(DurationOfPitch), Occupation,
                                         Gender, float(NumberOfFollowups), ProductPitched, float(PreferredPropertyStar),
                                         MaritalStatus, float(NumberOfTrips), int(Passport), int(PitchSatisfactionScore),
                                         int(OwnCar), Designation, float(MonthlyIncome), float(TotalVisiting)]],
                                       columns=['Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 
                                                'Occupation', 'Gender', 'NumberOfFollowups', 
                                                'ProductPitched', 'PreferredPropertyStar', 'MaritalStatus', 
                                                'NumberOfTrips', 'Passport', 'PitchSatisfactionScore', 
                                                'OwnCar', 'Designation', 'MonthlyIncome', 'TotalVisiting'])

            processed_data = preprocessor.transform(input_data)

            result = final_model.predict(processed_data)

            # Return result as "Book" or "Not Book"
            results = "Book" if result[0] == 1 else "Not Book"
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('home.html', results=f"Error during preprocessing: {e}")

    return render_template('home.html', results=results)  

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
