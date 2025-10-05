import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess_data(df):
    df.fillna(-999, inplace=True)
    df.replace("UNKNOWN", "Some", inplace=True)
    df.drop(["runtime", "filming_locations", "directors"], axis=1, inplace=True)
    for i in range(3):
        df.drop(f"actor_{i}_gender", axis=1, inplace=True)


def train_model_and_predict(train_file: str, test_file: str):
    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)
    y_train = df_train.pop("awards")
    preprocess_data(df_train)
    preprocess_data(df_test)

    mlb = MultiLabelBinarizer()
    for i in ["keywords", "genres"]:
        gen = mlb.fit_transform(df_train[i])
        gen_df = pd.DataFrame(gen, columns=mlb.classes_)
        df_train = pd.concat([df_train, gen_df], axis=1)
        test_gen = mlb.transform(df_test[i])
        test_gen_df = pd.DataFrame(test_gen, columns=mlb.classes_)
        df_test = pd.concat([df_test, test_gen_df], axis=1)
        df_train.drop([i], axis=1, inplace=True)
        df_test.drop([i], axis=1, inplace=True)

    cbr = CatBoostRegressor(
        n_estimators=1300,
        max_depth=5,
        learning_rate=0.15,
        train_dir="/tmp/catboost_info",
        logging_level="Silent",
        allow_writing_files=False,
    )
    cbr.fit(df_train.values, y_train)
    return cbr.predict(df_test.values)
