import joblib
import pandas as pd

# Loading the trained model here 
model = joblib.load('gsr_stress_model.pkl')

def predict_stress_level(gsr_value):
    # Prepare the input in the correct format for the model
    input_data = pd.DataFrame([[gsr_value]], columns=['Resistance (Ohms)'])

    # Getting the prediction
    prediction = model.predict(input_data)

    return prediction[0]

if __name__ == '__main__':
    ####################################################################
    # Enter the GSR value that we need here to predict the stress level
    test_value = 0.41023535985401216 # New value from our dataset "GSR_DATA-1.cleaned.csv line 455,0.41023535985401216,Remove 12 Pieces with Noise (1 min)"
    ####################################################################
    predicted_label = predict_stress_level(test_value)

    print(f"The input GSR value is: {test_value}")
    print(f"The model's prediction is: {predicted_label}")