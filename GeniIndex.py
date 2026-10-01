import pandas as pd
from pprint import pprint

data = [
    ["sunny", "hot", "high", "weak", "no"],
    ["sunny", "hot", "high", "strong", "no"],
    ["overcast", "hot", "high", "weak", "yes"],
    ["rainy", "mild", "high", "weak", "yes"],
    ["rainy", "cool", "normal", "weak", "yes"],
    ["rainy", "cool", "normal", "strong", "no"],
    ["overcast", "cool", "normal", "strong", "yes"],
    ["sunny", "mild", "high", "weak", "no"],
    ["sunny", "cool", "normal", "weak", "yes"],
    ["rainy", "mild", "normal", "weak", "yes"],
    ["sunny", "mild", "normal", "strong", "yes"],
    ["overcast", "mild", "high", "strong", "yes"],
    ["overcast", "hot", "normal", "weak", "yes"],
    ["rainy", "mild", "high", "strong", "no"],
]

columns = ["outlook", "temperature", "humidity", "wind", "play"]
df = pd.DataFrame(data, columns=columns)

def gini_impurity(sub_df, target_col="play"):
    total = len(sub_df)
    if total == 0:
        return 0

    counts = sub_df[target_col].value_counts()
    sum_sq_prob = 0.0

    for count in counts:
        prob = count / total
        sum_sq_prob += prob**2

    return 1 - sum_sq_prob

def gini_split_attribute(df, attribute, target_col="play"):
    total_samples = len(df)
    unique_values = df[attribute].unique()

    split_gini = 0.0
    details = {}
    for val in unique_values:
        sub_df = df[df[attribute] == val]
        val_gini = gini_impurity(sub_df, target_col)
        weight = len(sub_df) / total_samples

        split_gini += weight * val_gini
        details[val] = {
            "count": len(sub_df),
            "gini": val_gini,
            "distribution": dict(sub_df[target_col].value_counts()),
        }
    return split_gini, details

def build_tree_gini(df, features, target_col="play"):
    if len(df[target_col].unique()) == 1:
        return df[target_col].iloc[0]
    if not features:
        return df[target_col].mode()[0]
    best_feature = None
    min_gini = float("inf")

    for feature in features:
        g_split, _ = gini_split_attribute(df, feature, target_col)
        if g_split < min_gini:
            min_gini = g_split
            best_feature = feature
    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]

    for val in df[best_feature].unique():
        sub_df = df[df[best_feature] == val]
        tree[best_feature][val] = build_tree_gini(
            sub_df, remaining_features, target_col
        )
    return tree
if __name__ == "__main__":
    g_temp, temp_details = gini_split_attribute(df, "temperature")
    for val, info in temp_details.items():
        yes_cnt = info["distribution"].get("yes", 0)
        no_cnt = info["distribution"].get("no", 0)
        print(
            f"G({val}) = 1 - ({yes_cnt}/{info['count']})^2 - ({no_cnt}/{info['count']})^2 = {info['gini']:.4f}"
        )

    print(f"\nG_split(temperature) = {g_temp:.4f}\n")
    features = ["outlook", "temperature", "humidity", "wind"]
    for f in features:
        g_val, _ = gini_split_attribute(df, f)
        print(f"G_split({f:11s}) = {g_val:.4f}")

    print("\nCay quyet dinh xay bang thuat toan Geni Index")
    tree = build_tree_gini(df, features)
    pprint(tree)