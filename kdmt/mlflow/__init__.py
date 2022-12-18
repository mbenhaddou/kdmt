import mlflow
from kdmt.mlflow import _mlflow_utils, _model_utils, _utils
import os, tabulate
from kdmt.mlflow._list_model_versions import _get_model_versions
import pandas as pd
from mlflow.exceptions import RestException


#os.environ["MLFLOW_TRACKING_URI"] = "https://mentis.io/mlflow/"
try:
    client = mlflow.tracking.MlflowClient(os.environ['MLFLOW_TRACKING_URI'])
except Exception as e:
    raise Exception("Could not get server URI. t"
                    "try to set the MLFLOW_TRACKING_URI variable. "+ str(e))

TAG_PARENT_RUN_ID = "mlflow.parentRunId"
print("MLflow Version:", mlflow.version.VERSION)
print("MLflow Tracking URI:", mlflow.get_tracking_uri())

def get_best_run(experiment_id_or_name, metric, ascending=False, ignore_nested_runs=False):
    """
    Current search syntax does not allow to check for existence of a tag key so if there are nested runs we have to
    bring all runs back to the client making this costly and unscalable.
    """

    exp = _mlflow_utils.get_experiment(client, experiment_id_or_name)
    print("Experiment name:",exp.name)

    order_by = "ASC" if ascending else "DESC"
    column = "metrics.{} {}".format(metric, order_by)
    if ignore_nested_runs:
        runs = client.search_runs(exp.experiment_id, "", order_by=[column])
        runs = [ run for run in runs if TAG_PARENT_RUN_ID not in run.data.tags ]
    else:
        runs = client.search_runs(exp.experiment_id, "", order_by=[column], max_results=1)
    return runs[0].info.run_id,runs[0].data.metrics[metric]


def delete_model(model):
    _model_utils.delete_model(client, model)


def delete_model_stages(model, stages):
    print("Options:")
    for k,v in locals().items(): print(f"  {k}: {v}")
    stages = _utils.normalize_stages(stages)
    print(">> stages:",stages)
    versions = client.search_model_versions(f"name='{model}'")
    print(f"Found {len(versions)} versions for model {model}")
    for vr in versions:
        if len(stages) == 0 or vr.current_stage.lower() in stages:
            dct = { "version": vr.version, "stage": vr.current_stage, "run_id": vr.run_id }
            print(f"Deleting {dct}")
            client.delete_model_version(model, vr.version)


def __list(sort_attribute="name", verbose=False):
    exps = client.search_experiments()
    print("Found {} experiments".format(len(exps)))

    if sort_attribute == "name":
        exps = sorted(exps, key=lambda x: x.name)
    elif sort_attribute == "experiment_id":
        exps = sorted(exps, key=lambda x: int(x.experiment_id))

    if verbose:
        if sort_attribute == "lifecycle_stage":
            exps = sorted(exps, key=lambda x: x.lifecycle_stage)
        elif sort_attribute == "artifact_location":
            exps = sorted(exps, key=lambda x: x.artifact_location)
        list = [(exp.experiment_id, exp.name, exp.lifecycle_stage, exp.artifact_location) for exp in exps]
        df = pd.DataFrame(list, columns=["experiment_id", "name", "lifecycle_stage", "artifact_location"])
    else:
        list = [(exp.experiment_id, exp.name) for exp in exps]
        df = pd.DataFrame(list, columns=["experiment_id", "name"])

    print(tabulate(df, headers='keys', tablefmt='psql'))

    return df


def list_experiments(sort, verbose):
    print("Options:")
    for k, v in locals().items(): print(f"  {k}: {v}")
    __list(sort, verbose)


def list_model_versions(model, view, max_results):
    print("Options:")
    for k,v in locals().items(): print(f"  {k}: {v}")
    _get_model_versions(model, view, max_results)


def register_model(registered_model, run_id, model_artifact, stage):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(registered_model)
        print(f"Created new model '{registered_model}'")
    except RestException as e:
        if not "RESOURCE_ALREADY_EXISTS: Registered Model" in str(e):
            raise e
        print(f"Model '{registered_model}' already exists")

    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/{model_artifact}"
    print("Source:",source)

    version = client.create_model_version(registered_model, source, run_id)
    print("Version:",version)

    if stage:
        client.transition_model_version_stage(registered_model, version.version, stage)


def get_production_version(model_name, artifact_path=None):
    model=client.get_registered_model(model_name)
    for model_version in model.latest_versions:
        if model_version.current_stage=='Production':
            model_version_dict= dict(model_version)
            model_run=client.get_run(model_version.run_id)
            model_version_dict['metrics']=model_run.data.metrics
            return model_version_dict

    return None


def download_model(run_id, artifact_path, output_dir):
    """
    Downloads the model associated with the model URI.
    - For model scheme, downloads the model associated with the stage/version.
    - For run scheme, downloads the model associated with the run ID and artifact.
    :param: model_uri - MLflow model URI.
    :param:output_dir - Output directory for downloaded model.
    :return: The local path to the downloaded model.
    """
    return client.download_artifacts(run_id, artifact_path, dst_path=output_dir)


def is_model_registered(model_name):
    return model_name in  [m.name for m in client.search_registered_models()]

if __name__ == "__main__":
    import mlflow
    import os

    model=is_model_registered("SklearnEstimator")
    print(model)

    best_run=get_best_run('email_signature', "F1")

    download_model(best_run[0], 'current', ".")

#    get_best_run("email_signature",
#    "F1")
    #    mlflow.set_tracking_uri("https://mentis.io/mlflow/")
#    client =mlflow.MlflowClient(tracking_uri="https://mentis.io/mlflow/")
#    for rm in client.search_registered_models():
#       print(dict(rm))