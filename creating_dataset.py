import pandas as pd
from dga import *

# get column with domains for future 10000 legitimate_domain_data sets
majestic_million_data = pd.read_csv("legitimate_domain_data/majestic_million.csv", nrows=10000)
legitimate_domain_data = majestic_million_data.iloc[:, [2]]
legitimate_domain_data.to_csv("legitimate_domain_data/legitimate_domain.csv", index=False)

# delete duplicates in train
val_data = pd.read_csv("val.csv")
test_data = pd.read_csv("test.csv")
unfiltered_legitimate_train_data = pd.read_csv("legitimate_domain_data/legitimate_domain.csv")

exclude_data = set(val_data.iloc[:, 0]).union(set(test_data.iloc[:, 0]))
legitimate_train_data = unfiltered_legitimate_train_data[
    ~unfiltered_legitimate_train_data.iloc[:, 0].isin(exclude_data)]
legitimate_train_data.to_csv("legitimate_domain_data/legitimate_domain.csv", index=False)

# take 10000 legitimate_domain_data sets for legitimate domains & set is_dga
legitimate_train_data_part = pd.read_csv("legitimate_domain_data/legitimate_domain.csv", nrows=10000)
legitimate_train_data_part.columns = ["domain"]
legitimate_train_data_part["is_dga"] = 0

# add dga domains & set is_dga
dga_domains = generate_dga_domains(10000, 5, 20)
dga_df = pd.DataFrame(dga_domains, columns=['domain'])
dga_df["is_dga"] = 1

# combine legitimate and dga domains & shuffle
combined_data = pd.concat([legitimate_train_data_part, dga_df], ignore_index=True)
combined_data = combined_data.sample(frac=1).reset_index(drop=True)
combined_data.to_csv("dataset.csv", index=False)
