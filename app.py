from flask import Flask, escape, request, render_template

import pickle

model = pickle.load(open("RandomForest_reg_model.pkl", 'rb'))

app = Flask(__name__)


@app.route("/analysis")
def analysis():
    return render_template("cars.html")


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        km_driven = int(request.form['km_driven'])
        fuel = request.form['fuel']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        year = int(request.form['year'])

        # fuel
        if(fuel == "Diesel"):
            fuel_Diesel = 1
            fuel_LPG = 0
            fuel_Petrol = 0
        elif(fuel == "LPG"):
            fuel_Diesel = 0
            fuel_LPG = 1
            fuel_Petrol = 0
        elif(fuel == "Petrol"):
            fuel_Diesel = 0
            fuel_LPG = 0
            fuel_Petrol = 1
        else:
            fuel_Diesel = 0
            fuel_LPG = 0
            fuel_Petrol = 0

        # seller_type
        if(seller_type == "Individual"):
            seller_type_Individual = 1
            seller_type_Trustmark_Dealer = 0
        elif(seller_type == "Trustmark Dealer"):
            seller_type_Individual = 0
            seller_type_Trustmark_Dealer = 1
        else:
            seller_type_Individual = 0
            seller_type_Trustmark_Dealer = 0

        # transmission
        if(transmission == "Manual"):
            transmission_Manual = 1
        else:
            transmission_Manual = 0

        # owner
        if(owner == "Second Owner"):
            owner_Second_Owner = 1
            owner_Fourth_and_Above_Owner = 0
            owner_Third_Owner = 0
            owner_Test_Drive_Car = 0
        elif(owner == "Fourth & Above Owner"):
            owner_Second_Owner = 0
            owner_Fourth_and_Above_Owner = 1
            owner_Third_Owner = 0
            owner_Test_Drive_Car = 0
        elif(owner == "Third Owner"):
            owner_Second_Owner = 0
            owner_Fourth_and_Above_Owner = 0
            owner_Third_Owner = 1
            owner_Test_Drive_Car = 0
        elif(owner == "Test Drive Car"):
            owner_Second_Owner = 0
            owner_Fourth_and_Above_Owner = 0
            owner_Third_Owner = 0
            owner_Test_Drive_Car = 1
        else:
            owner_Second_Owner = 0
            owner_Fourth_and_Above_Owner = 0
            owner_Third_Owner = 0
            owner_Test_Drive_Car = 0

        year = 2021 - year

        predition = model.predict([[km_driven, year, fuel_Diesel, fuel_LPG,
                                    fuel_Petrol, seller_type_Individual, seller_type_Trustmark_Dealer,
                                    transmission_Manual, owner_Fourth_and_Above_Owner,
                                    owner_Second_Owner, owner_Test_Drive_Car, owner_Third_Owner]])

        output = round(predition[0], 2)

        if output < 0:
            return render_template('index.html',prediction_text="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell This Car at {}".format(output))

    else:
        return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)
