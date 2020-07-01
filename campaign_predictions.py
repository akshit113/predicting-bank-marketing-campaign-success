from pprint import pprint

from pandas import read_csv, DataFrame, concat, get_dummies


def import_dataset(fpath):
    data = read_csv(fpath)
    print(data.head())
    print(data.shape)
    return data


def create_age_buckets(df):
    print(df['age'].min())
    print(df['age'].max())
    old_values = df['age'].values.tolist()
    new_age_list = []
    new_age = 0
    for age in old_values:
        if age in range(18, 30):
            new_age = 1
        elif age in range(30, 40):
            new_age = 2
        elif age in range(40, 50):
            new_age = 3
        elif age in range(50, 65):
            new_age = 4
        elif age in range(65, 70):
            new_age = 5
        else:
            new_age = 6
        new_age_list.append(new_age)
    age_df = DataFrame(new_age_list, columns=['Age_Buckets'])
    df = concat([age_df, df], axis=1)
    return df


def check_unique_value(df, colnames):
    """Gets unique value counts for all selected columns in the dataframe including NaN values and PrettyPrints the
    dicitonary
    :param df:
    :param colnames:
    :return: a dictionary
    """
    mydict = {}
    for col in colnames:
        val_count = (df[col].value_counts(dropna=False)).to_dict()
        mydict[col] = val_count
    pprint(mydict)
    return


def normalize_columns(df, colnames):
    for col in colnames:
        s = df[col]
        df[col] = s.sub(s.min()).div((s.max() - s.min()))
    print(f'''Normalized Columns: {colnames}''')

    return df


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """
    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    return df


def clean_dataset(df):
    df = create_age_buckets(df)
    print(df.columns)
    check_unique_value(df, ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome',
                            'y'])
    df = normalize_columns(df, ['balance', 'day', 'duration', 'campaign', 'pdays'])
    df = one_hot_encode(df, ['job', 'marital', 'education', 'default', 'housing',
                             'loan', 'contact', 'month', 'poutcome', 'y'])
    return df


def main():
    fpath = r'C:\Users\akshitagarwal\Desktop\Keras\datasets\bank\bank\bank_data.csv'
    data = import_dataset(fpath)
    print(data.isna().sum())
    # no NA values found in the dataset
    df = clean_dataset(data)
    print('done')


if __name__ == '__main__':
    main()
