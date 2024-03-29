import os
try:
    import mlflow
except:
    pass


def dump_mlflow_info():
    """ Show basic MLflow information. """
    print("MLflow Info:")
    print("  MLflow Version:", mlflow.version.VERSION)
    print("  Tracking URI:", mlflow.tracking.get_tracking_uri())
    mlflow_host = get_mlflow_host()
    print("  Real MLflow host:", mlflow_host)
    print("  MLFLOW_TRACKING_URI:", os.environ.get("MLFLOW_TRACKING_URI",""))
    print("  DATABRICKS_HOST:", os.environ.get("DATABRICKS_HOST",""))
    print("  DATABRICKS_TOKEN:", os.environ.get("DATABRICKS_TOKEN",""))


def get_mlflow_host():
    """ Returns the host (tracking URI) and token."""
    return get_mlflow_host_token()[0]

def get_mlflow_client():
    if "MLFLOW_TRACKING_URI" not in os.environ:
        raise Exception("Could not get server URI. try to set the MLFLOW_TRACKING_URI variable. ")

    try:
        client = mlflow.tracking.MlflowClient(os.environ['MLFLOW_TRACKING_URI'])

        print("MLflow Version:", mlflow.version.VERSION)
        print("MLflow Tracking URI:", mlflow.get_tracking_uri())
    except Exception as e:
        raise Exception("Cannot conect to mlflow client, raised Exception: " + str(e))

    return client


def get_mlflow_host_token():
    """ Returns the host (tracking URI) and token. """
    uri = os.environ.get('MLFLOW_TRACKING_URI',None)
    if uri is not None and uri != "databricks":
        return (uri,None)



def get_experiment(client, exp_id_or_name):
    """
    Gets an experiment either by ID or name.
    :param: client - MLflowClient.
    :param: exp_id_or_name - Experiment ID or name..
    :return: Experiment object.
    """
    exp = client.get_experiment_by_name(exp_id_or_name)
    if exp is None:
        try:
            exp = client.get_experiment(exp_id_or_name)
        except Exception:
            raise Exception(f"Cannot find experiment ID or name '{exp_id_or_name}'. Client: {client}'")
    return exp


def get_last_run(mlflow_client, exp_id_or_name):
    exp = get_experiment(mlflow_client, exp_id_or_name)
    runs = mlflow_client.search_runs(exp.experiment_id, order_by=["attributes.start_time desc"], max_results=1)
    return runs[0]
