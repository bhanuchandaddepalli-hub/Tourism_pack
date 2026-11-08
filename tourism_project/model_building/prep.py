import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv") #Load Dataset
df.drop(columns=["CustomerID", "Unnamed: 0"], 
        errors="ignore", inplace=True)  #Removing Unnecessary columns

# Data cleaning - Categorical Var

# Standardize Gender values
df["Gender"] = df["Gender"].str.strip().str.lower()

df["Gender"] = df["Gender"].replace({
    "fe male": "Female",
    "female": "Female",
    "male": "Male"
})

# Standardize MaritalStatus values
df["MaritalStatus"] = df["MaritalStatus"].str.strip().str.lower()

df["MaritalStatus"] = df["MaritalStatus"].replace({
    "single": "Single",
    "unmarried": "Single",
    "married": "Married",
    "divorced": "Divorced"
})

# Verify corrected categorical values
print("Gender distribution:")
print(df["Gender"].value_counts())

print("\nMarital Status distribution:")
print(df["MaritalStatus"].value_counts())

# Separate Features and Target

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Train-Test Split

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["Type"].unique()))
print("\nTraining target distribution:")
print(ytrain.value_counts())

print("\nTraining target percentage:")
print((ytrain.value_counts(normalize=True) * 100).round(2))
