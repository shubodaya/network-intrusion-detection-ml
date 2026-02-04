import numpy as np
import pandas as pd


def get_feature_names(preprocessor):
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "steps"):
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, "get_feature_names_out"):
            try:
                names = transformer.get_feature_names_out(cols)
            except TypeError:
                names = transformer.get_feature_names_out()
            feature_names.extend(list(names))
        elif hasattr(transformer, "get_feature_names"):
            try:
                names = transformer.get_feature_names(cols)
            except TypeError:
                names = transformer.get_feature_names()
            feature_names.extend(list(names))
        else:
            if isinstance(cols, (list, tuple, np.ndarray, pd.Index)):
                feature_names.extend(list(cols))
            else:
                feature_names.append(cols)
    return feature_names


def extract_feature_importance(pipeline):
    if not hasattr(pipeline, "named_steps"):
        return None, None
    preprocessor = pipeline.named_steps.get("preprocess")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        return None, None

    feature_names = get_feature_names(preprocessor)
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    if importances is None:
        return None, None

    if len(feature_names) != len(importances):
        return None, None
    return feature_names, importances

