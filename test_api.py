import requests
import json

# URL de la API
url = 'http://localhost:5002/predict'

# Datos de ejemplo para una reclamación legítima
legitimate_claim = {
    "Customer_Age": 35,
    "Gender": "M",
    "Insured_MaritalStatus": "Casado",
    "Insured_Occupation": "Empleado Privado",
    "Insured_Zip": 28001,
    "Insured_Inception_Date": "2018-01-01",
    "Policy_Start_Date": "2023-01-01",
    "Last_Purchase_History_Date": "2023-01-01",
    "Coverage_description": "Responsabilidad Civil",
    "Coverage_Amount": 50000,
    "Premium_Amount": 1200,
    "Beneficiary_Type_Description": "Asegurado",
    "Claim_History_Count_This_Policy": 0,
    "Claim_Frequency_Last_12_Month": 0,
    "Vehicle_Make": "Toyota",
    "Vehicle_Model": "Corolla",
    "Model_Year": 2020,
    "Incident_Date": "2023-06-15",
    "Date_Reported": "2023-06-16",
    "Claim_Amount": 3000,
    "LossType_Description": "Pérdida Parcial",
    "Branch_Description": "Punta Cana",
    "WorkShop_Name": "Taller Los Prados",
    "Claim_Description": "Tuve un accidente en la autopista cuando un vehículo me cerró el paso. El impacto fue en la parte trasera izquierda."
}

# Datos de ejemplo para una reclamación fraudulenta
fraudulent_claim = {
    "Customer_Age": 25,
    "Gender": "M",
    "Insured_MaritalStatus": "Soltero",
    "Insured_Occupation": "Desempleado",
    "Insured_Zip": 28001,
    "Insured_Inception_Date": "2022-01-01",
    "Policy_Start_Date": "2023-01-01",
    "Last_Purchase_History_Date": "2023-01-01",
    "Coverage_description": "Incendio",
    "Coverage_Amount": 20000,
    "Premium_Amount": 400,
    "Beneficiary_Type_Description": "Asegurado",
    "Claim_History_Count_This_Policy": 3,
    "Claim_Frequency_Last_12_Month": 2,
    "Vehicle_Make": "Honda",
    "Vehicle_Model": "Civic",
    "Model_Year": 2015,
    "Incident_Date": "2023-06-01",
    "Date_Reported": "2023-06-10",
    "Claim_Amount": 15000,
    "LossType_Description": "Pérdida Total",
    "Branch_Description": "Oficina Principal",
    "WorkShop_Name": "Auto Pintura J&J",
    "Claim_Description": "El vehículo sufrió daños inexplicables. Necesito el dinero urgentemente para comprar un auto nuevo."
}

def test_api(claim_data, description):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(claim_data), headers=headers)

    if response.status_code == 200:
        result = response.json()
        print(f"\n{description}:")
        print(f"Probabilidad de Fraude: {result['fraud_probability']:.4f}")
        print(f"Predicción: {result['prediction']}")
    else:
        print(f"Error en {description}: {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("Probando API de Detección de Fraude...")
    test_api(legitimate_claim, "Reclamación Legítima")
    test_api(fraudulent_claim, "Reclamación Fraudulenta")