import numpy as np
import pandas as pd
import tldextract


def domain_extract(url):
    ext = tldextract.extract(url)
    if not ext.suffix:
        return np.nan
    else:
        return ext.domain


train_data = pd.read_csv('train.csv')

# delete tld
train_data['domain'] = [domain_extract(url) for url in train_data['domain']]

# delete duplicates
train_data = train_data.drop_duplicates()

# process na & empty fields
# delete na & empty domains
train_data = train_data.dropna(subset=['domain'])
train_data = train_data[train_data['domain'] != ""]
train_data = train_data.reset_index(drop=True)
# fill na is_dga with mode
train_data['is_dga'] = train_data['is_dga'].fillna(train_data['is_dga'].mode()[0])

train_data.to_csv('train.csv', index=False)
