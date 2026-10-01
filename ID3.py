import math
import pandas as pd
from pprint import pprint

data = [
    ["<=30", "high", "no", "fair", "no"],
    ["<=30", "high", "no", "excellent", "no"],
    ["31...40", "high", "no", "fair", "yes"],
    [">40", "medium", "no", "fair", "yes"],
    [">40", "low", "yes", "fair", "yes"],
    [">40", "low", "yes", "excellent", "no"],
    ["31...40", "low", "yes", "excellent", "yes"],
    ["<=30", "medium", "no", "fair", "no"],
    ["<=30", "low", "yes", "fair", "yes"],
    [">40", "medium", "yes", "fair", "yes"],
    ["<=30", "medium", "yes", "excellent", "yes"],
    ["31...40", "medium", "no", "excellent", "yes"],
    ["31...40", "high", "yes", "fair", "yes"],
    [">40", "medium", "no", "excellent", "no"],
]

columns = ["age", "income", "student", "credit_rating", "buys_computer"]
df = pd.DataFrame(data, columns=columns)

def calculate_entropy(df, target_attribute="buys_computer"):
    target_counts = df[target_attribute].value_counts()
    total_samples = len(df)
    entropy = 0.0
    for count in target_counts:
        probability = count / total_samples
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_information_gain(df, attribute, target_attribute="buys_computer"):
    total_entropy = calculate_entropy(df, target_attribute)
    total_samples = len(df)
    feature_values = df[attribute].unique()
    weighted_entropy = 0.0

    for value in feature_values:
        sub_df = df[df[attribute] == value]
        prob = len(sub_df) / total_samples
        weighted_entropy += prob * calculate_entropy(sub_df, target_attribute)
    return total_entropy - weighted_entropy

def id3(df, features, target_attribute="buys_computer"):
    unique_targets = df[target_attribute].unique()
    if len(unique_targets) == 1:
        return unique_targets[0]
    if len(features) == 0:
        return df[target_attribute].mode()[0]
    gains = {
        feat: calculate_information_gain(df, feat, target_attribute)
        for feat in features
    }
    best_feature = max(gains, key=gains.get)
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    # Phân nhánh theo các giá trị của thuộc tính tốt nhất
    for value in df[best_feature].unique():
        sub_df = df[df[best_feature] == value]

        if len(sub_df) == 0:
            tree[best_feature][value] = df[target_attribute].mode()[0]
        else:
            tree[best_feature][value] = id3(
                sub_df, remaining_features, target_attribute
            )

    return tree

def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree

    root_node = next(iter(tree))
    feature_value = sample.get(root_node)

    if feature_value in tree[root_node]:
        return predict(tree[root_node][feature_value], sample)
    else:
        return "Khong xac dinh"

if __name__ == "__main__":
    feature_names = ["age", "income", "student", "credit_rating"]
    decision_tree = id3(df, feature_names)
    print("Cay quyet dinh xay dung bang thuat toan ID3")
    pprint(decision_tree)
    test_sample = {
        "age": "<=30",
        "income": "medium",
        "student": "yes",
        "credit_rating": "fair",
    }
    result = predict(decision_tree, test_sample)
    print(f"\nKet qua du doan cho mau {test_sample}: buys_computer = {result}")