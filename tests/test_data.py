from street_sign_project.utils import project_root

train_label_path = project_root() / "data" / "preprocessed" / "train" / "labels"
test_label_path = project_root() / "data" / "preprocessed" / "test" / "labels"
val_label_path = project_root() / "data" / "preprocessed" / "valid" / "labels"

train_counts = [0] * 73
test_counts = [0] * 73
val_counts = [0] * 73
overall_counts = [0] * 73

for train_label in train_label_path.glob("*.txt"):
    with train_label.open() as file:
        for line in file:
            values = line.strip().split()
            if not values or len(values) != 5:
                continue
            train_counts[int(values[0])] += 1

for test_label in test_label_path.glob("*.txt"):
    with test_label.open() as file:
        for line in file:
            values = line.strip().split()
            if not values or len(values) != 5:
                continue
            test_counts[int(values[0])] += 1

for val_label in val_label_path.glob("*.txt"):
    with val_label.open() as file:
        for line in file:
            values = line.strip().split()
            if not values or len(values) != 5:
                continue
            val_counts[int(values[0])] += 1

overall_counts = [x + y + z for x, y, z in zip(train_counts, test_counts, val_counts)]
train_props = [round(x / y, 3) for x, y in zip(train_counts, overall_counts)]
test_props = [round(x / y, 3) for x, y in zip(test_counts, overall_counts)]
val_props = [round(x / y, 3) for x, y in zip(val_counts, overall_counts)]
print(
    f"training class id {train_props.index(max(train_props))} with {max(train_props)} of {overall_counts[train_props.index(max(train_props))]} occurances"
)
print(
    f"training class id {train_props.index(min(train_props))} with {min(train_props)} of {overall_counts[train_props.index(min(train_props))]} occurances"
)
print(
    f"test class id {test_props.index(max(test_props))} with {max(test_props)} of {overall_counts[test_props.index(max(test_props))]} occurances"
)
print(
    f"test class id {test_props.index(min(test_props))} with {min(test_props)} of {overall_counts[test_props.index(min(test_props))]} occurances"
)
print(
    f"validation class id {val_props.index(max(val_props))} with {max(val_props)} of {overall_counts[val_props.index(max(val_props))]} occurances"
)
print(
    f"validation class id {val_props.index(min(val_props))} with {min(val_props)} of {overall_counts[val_props.index(min(val_props))]} occurances"
)
