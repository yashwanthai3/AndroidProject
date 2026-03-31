from androguard.misc import AnalyzeAPK
import numpy as np

# Example feature list (must match training features)
IMPORTANT_PERMISSIONS = [
    "android.permission.SEND_SMS",
    "android.permission.READ_SMS",
    "android.permission.INTERNET",
    "android.permission.READ_CONTACTS",
    "android.permission.ACCESS_FINE_LOCATION",
]

def extract_features(apk_path):

    a, d, dx = AnalyzeAPK(apk_path)
    permissions = a.get_permissions()

    feature_vector = []

    for perm in IMPORTANT_PERMISSIONS:
        if perm in permissions:
            feature_vector.append(1)
        else:
            feature_vector.append(0)

    return np.array(feature_vector)