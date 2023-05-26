from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle

# load model
model = pickle.load(open("model/customer_behaver_predictor.pkl", 'rb'))

app = FastAPI()


class Input(BaseModel):
    hour: int
    age: int
    numscreens: int
    minigame: int
    used_premium_feature: int
    location: int
    Institutions: int
    VerifyPhone: int
    BankVerification: int
    VerifyDateOfBirth: int
    ProfilePage: int
    VerifyCountry: int
    idscreen: int
    Splash: int
    Finances: int
    Alerts: int
    VerifyMobile: int
    VerifyHousing: int
    VerifyHousingAmount: int
    Rewards: int
    AccountView: int
    VerifyAnnualIncome: int
    Login: int
    WebView: int
    SecurityModal: int
    ResendToken: int
    TransactionList: int
    Other: int
    SavingCount: int
    CMCount: int
    CCCount: int
    LoansCount: int


@app.get("/")
def read_root():
    return {"msg": "Customer Behavior Predictor"}


@app.post("/predict")
def predict_customer_behavior(input: Input):
    data = input.dict()
    data_in = [[data['hour'], data['age'], data['numscreens'], data['minigame'],
                data['used_premium_feature'], data['location'], data['Institutions'],
                data['VerifyPhone'], data['BankVerification'], data['VerifyDateOfBirth'],
                data['ProfilePage'], data['VerifyCountry'], data['idscreen'],
                data['Splash'], data['Finances'], data['Alerts'],
                data['VerifyMobile'], data['VerifyHousing'], data['VerifyHousingAmount'],
                data['Rewards'], data['AccountView'], data['VerifyAnnualIncome'],
                data['Login'], data['WebView'], data['SecurityModal'],
                data['ResendToken'], data['TransactionList'], data['Other'],
                data['SavingCount'], data['CMCount'], data['CCCount'],
                data['LoansCount']]]

    prediction = model.predict(data_in)
    return {
        'prediction': prediction[0]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
