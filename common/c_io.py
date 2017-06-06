from sklearn.externals import joblib
import pathlib


def persist(file_name,
            object_to_be_persisted):

    joblib.dump(object_to_be_persisted, file_name)


def read(file_name):
    """
    Returns
    -------
    (file_exists, loaded_object):
    """
    if not exists(file_name):
        return False, None

    return True, joblib.load(file_name)


def exists(file_name):

    path = pathlib.Path(file_name)
    return path.is_file()
