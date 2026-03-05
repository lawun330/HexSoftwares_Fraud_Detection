import pandas as pd

# load full dataset
df = pd.read_csv("../data/creditcard.csv")

# separate fraud and non-fraud
fraud = df[df["Class"] == 1]
non_fraud = df[df["Class"] == 0]

# sample non-fraud rows
n_non_fraud = 4920
non_fraud_sample = non_fraud.sample(n=n_non_fraud, random_state=42)

# combine all fraud and sampled non-fraud
sample_df = pd.concat([fraud, non_fraud_sample], axis=0)

# shuffle the resulting dataframe
sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)

# save to CSV
sample_df.to_csv("../data/creditcard_sample.csv", index=False)

print("Original shape:\t", df.shape)
print("Sample shape:\t", sample_df.shape)
print("Class distribution in sample:")
print(sample_df["Class"].value_counts())