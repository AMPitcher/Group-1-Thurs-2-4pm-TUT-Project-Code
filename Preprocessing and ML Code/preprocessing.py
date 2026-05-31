import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder
from sklearn.model_selection import train_test_split
import re
import numpy as np

SEED = 42

southern_hemisphere = {
    "Brazil",
    "Argentina",
    "Australia",
    "Madagascar",
    "South Africa",
    "Tanzania",
    "Kenya",
    "Zambia",
    "Zimbabwe",
    "Malawi",
    "New Zealand",
    "Chile",
    "Burundi",
    "Congo DRC",
    "Uganda",
    "Mozambique",
}

equatorial_countries = {
    "Singapore",
    "Indonesia",
    "Malaysia",
    "Nigeria",
    "Cameroon",
    "Sierra Leone",
    "Liberia",
    "Uganda",
    "Kenya",
    "Ethiopia",
}


def clean_multilabel(entry):
    if pd.isna(entry):
        return []
    entry = str(entry)
    entry = re.sub(r"([a-z])([A-Z])", r"\1,\2", entry)
    entry = re.sub(r"(cover)(Other)", r"\1,\2", entry)
    items = entry.split(",")
    cleaned = []
    for item in items:
        item = item.strip()
        if item.lower() in ["nan", "none", "Other", "none, other", "other blocks"]:
            continue
        if "other" in item.lower():
            continue
        cleaned.append(item)
    return cleaned


def consolidate_veg(label_list):
    veg_map = {
        "Grass": "Grass",
        "Grass_small": "Grass",
        "Grass_small_plants": "Grass",
        "Concrete_Road": "Concrete_impermeable",
        "Concrete_impermeable_surface": "Concrete_impermeable",
        "No_vegetation_cover": "No_vegetation",
        "Bare_soil": "No_vegetation",
        "Bare_rock_sand": "No_vegetation",
        "Trees_shrubs": "Trees_shrubs",
        "None_observed": "None_observed",
    }
    return list(set(veg_map.get(l, l) for l in label_list))


def consolidate_aqua(label_list):
    aqua_map = {
        "Floating_plants": "Aquatic_plants",
        "Plants_below_surface": "Aquatic_plants",
        "Plants_emerging_from_water": "Aquatic_plants",
        "Fish": "Sensitive_bioindicators",
        "Dragonflies_damselflies": "Sensitive_bioindicators",
        "Frogs_toads": "Sensitive_bioindicators",
        "Aquatic_birds": "Generalist_animals",
        "Reptiles": "Generalist_animals",
        "Aquatic_mammals": "Generalist_animals",
        "None_observed": "None_observed",
    }
    return list(set(aqua_map.get(l, l) for l in label_list))


def fix_no_vegetation(label_list):
    actual = {"Trees_shrubs", "Grass", "Concrete_impermeable"}
    if any(l in actual for l in label_list):
        return [l for l in label_list if l != "No_vegetation"]
    return label_list


def get_region(row):
    country = str(row["Country"])
    if country in equatorial_countries:
        return "Equatorial"
    elif country in southern_hemisphere:
        return "Southern"
    return "Northern"


def get_season(row):
    month = row["month"]
    country = str(row["Country"])
    if pd.isna(month):
        return None
    if country in equatorial_countries:
        return "Wet/Dry (Equatorial)"
    elif country in southern_hemisphere:
        if month in [12, 1, 2]:
            return "Summer"
        elif month in [3, 4, 5]:
            return "Autumn"
        elif month in [6, 7, 8]:
            return "Winter"
        elif month in [9, 10, 11]:
            return "Spring"
    else:
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Autumn"
    return None


def load_and_prepare(csv_path="Global_Data_Set_XvsX_0.csv"):
    observable_features = [
        "Freshwater body type",
        "What is the main land use within 50m?",
        "What is the main bank vegetation? (select all that apply)",
        "What aquatic life is there evidence of? (select all that apply)",
        "Which of these best describes the dominant form of algae present?",
        "Estimate the water flow",
        "Estimate the water level",
        "Estimate the water colour",
        "Sample Date",
        "Country",
    ]
    numerical_features = [
        "sort_order_nitrates",
        "sort_order_phosphates",
        "sort_order_turbidity",
    ]
    target = ["Feedback Rating"]

    df = pd.read_csv(
        csv_path, usecols=observable_features + numerical_features + target
    )

    # Fix data types
    for col in ["sort_order_nitrates", "sort_order_phosphates", "sort_order_turbidity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Sample Date"] = pd.to_datetime(df["Sample Date"], errors="coerce")
    for col in [
        "Freshwater body type",
        "What is the main land use within 50m?",
        "Which of these best describes the dominant form of algae present?",
        "Estimate the water flow",
        "Estimate the water level",
        "Estimate the water colour",
        "Feedback Rating",
        "Country",
    ]:
        df[col] = df[col].astype("category")
    for col in [
        "What is the main bank vegetation? (select all that apply)",
        "What aquatic life is there evidence of? (select all that apply)",
    ]:
        df[col] = df[col].astype(str)

    # Clean multi-label columns
    for col in [
        "What is the main bank vegetation? (select all that apply)",
        "What aquatic life is there evidence of? (select all that apply)",
    ]:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True).fillna("None")

    df["bank_veg_list"] = df[
        "What is the main bank vegetation? (select all that apply)"
    ].apply(clean_multilabel)
    df["aquatic_life_list"] = df[
        "What aquatic life is there evidence of? (select all that apply)"
    ].apply(clean_multilabel)
    df["bank_veg_list"] = df["bank_veg_list"].apply(
        lambda l: l if len(l) > 0 else ["None_observed"]
    )
    df["aquatic_life_list"] = df["aquatic_life_list"].apply(
        lambda l: l if len(l) > 0 else ["None_observed"]
    )

    df["bank_veg_list"] = (
        df["bank_veg_list"].apply(consolidate_veg).apply(fix_no_vegetation)
    )
    df["aquatic_life_list"] = df["aquatic_life_list"].apply(consolidate_aqua)

    df = df.drop(
        columns=[
            "What is the main bank vegetation? (select all that apply)",
            "What aquatic life is there evidence of? (select all that apply)",
        ]
    )

    # Clean target
    df = df[~df["Feedback Rating"].isin(["No water available", "Unknown"])]
    df["ecological_status"] = (
        df["Feedback Rating"].str.lower().str.strip().astype("category")
    )
    df = df.dropna(subset=["ecological_status"]).drop(columns=["Feedback Rating"])

    # Clean other columns
    rare_water = {"Wetland", "Canal", "Ditch", "Other"}
    df["Freshwater body type"] = (
        df["Freshwater body type"]
        .apply(lambda x: "Other" if x in rare_water else x)
        .astype("category")
    )

    land_use_map = {
        "Agriculture": "Agriculture",
        "Mixed_agricultural": "Agriculture",
        "Mixed agricultural": "Agriculture",
        "Arable_agricultural": "Agriculture",
        "Livestock": "Agriculture",
        "Agriculture_livestock": "Agriculture",
        "Grassland_shrubs": "Grassland_shrub",
        "Forest_plantation": "Forest",
        "Industrial": "Industrial_commercial",
        "Moorland_peat_bog": "Grassland_shrub",
        "Urban_residential": "Urban_residential",
        "Urban_green_space": "Urban_green_space",
        "Grassland_shrub": "Grassland_shrub",
        "Rural_residential": "Rural_residential",
        "Other": "Other",
        "Forest": "Forest",
        "Industrial_commercial": "Industrial_commercial",
    }
    df["What is the main land use within 50m?"] = df[
        "What is the main land use within 50m?"
    ].map(land_use_map)
    df = df.dropna(subset=["What is the main land use within 50m?"])
    df["What is the main land use within 50m?"] = df[
        "What is the main land use within 50m?"
    ].astype("category")

    df["Which of these best describes the dominant form of algae present?"] = df[
        "Which of these best describes the dominant form of algae present?"
    ].fillna("No_algae")

    df = df.dropna(
        subset=[
            "Estimate the water flow",
            "Estimate the water level",
            "Estimate the water colour",
        ]
    )
    df["Estimate the water level"] = df[
        "Estimate the water level"
    ].cat.remove_unused_categories()

    # Feature engineering
    df["Country_region"] = df.apply(get_region, axis=1).astype("category")
    df["month"] = df["Sample Date"].dt.month
    df["Season"] = df.apply(get_season, axis=1).astype("category")
    df = df.dropna(subset=["Season"]).drop(columns=["Sample Date", "month", "Country"])

    df = df.rename(
        columns={
            "Freshwater body type": "water_type",
            "What is the main land use within 50m?": "land_use_50m",
            "Which of these best describes the dominant form of algae present?": "algae_present",
            "Estimate the water flow": "water_flow",
            "Estimate the water level": "water_level",
            "Estimate the water colour": "water_colour",
        }
    )

    # One-hot encode multi-label columns
    mlb_veg = MultiLabelBinarizer()
    veg_encoded = pd.DataFrame(
        mlb_veg.fit_transform(df["bank_veg_list"]),
        columns=["veg_" + c for c in mlb_veg.classes_],
        index=df.index,
    )
    mlb_aqua = MultiLabelBinarizer()
    aqua_encoded = pd.DataFrame(
        mlb_aqua.fit_transform(df["aquatic_life_list"]),
        columns=["aqua_" + c for c in mlb_aqua.classes_],
        index=df.index,
    )
    df = df.drop(columns=["bank_veg_list", "aquatic_life_list"])
    df = pd.concat([df, veg_encoded, aqua_encoded], axis=1)

    for col in df.select_dtypes("category").columns:
        df[col] = df[col].cat.remove_unused_categories()

    # Train/test split
    y_raw = df["ecological_status"]
    df = df.drop(
        columns=["sort_order_nitrates", "sort_order_phosphates", "sort_order_turbidity"]
    )
    X = df.drop(columns=["ecological_status"])

    oe = OrdinalEncoder(categories=[["poor", "moderate", "good"]])
    y = oe.fit_transform(y_raw.values.reshape(-1, 1)).ravel().astype(int)

    cat_cols = X.select_dtypes("category").columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X_train, X_test, y_train, y_test


load_and_prepare(csv_path="Global_Data_Set_XvsX_0.csv")
