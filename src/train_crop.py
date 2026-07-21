import sys; sys.path.insert(0, 'src')
from preprocess import load_crop_data
import pickle, numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train():
    X, y, feature_names = load_crop_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.01,
        subsample=1.0, colsample_bytree=0.7,
        min_child_weight=5, gamma=0,
        reg_alpha=1, reg_lambda=1.5,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, tree_method='hist')

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=50)

    preds = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds))

    pickle.dump(model, open('models/xgb_crop.pkl', 'wb'))
    print("Saved: models/xgb_crop.pkl")

    importances = sorted(zip(feature_names, model.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    print("\nFeature importances:")
    for name, score in importances:
        print(f" {name}: {score:.4f}")


if __name__ == '__main__':
    train()
