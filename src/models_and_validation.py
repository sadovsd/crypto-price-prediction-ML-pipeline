import pandas as pd
from sklearn.metrics import precision_score



def get_precision_score_via_backtest(data, model, predictors, start=2500, step=200):
    
    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index, name="predictions")
        combined = pd.concat([test["target"], preds], axis=1)
        return combined

    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    data_with_predictions = pd.concat(all_predictions)

    return precision_score(data_with_predictions["target"], data_with_predictions["predictions"])
