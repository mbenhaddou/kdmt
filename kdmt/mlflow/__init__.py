try:
    import mlflow
    from kdmt.mlflow import _mlflow_utils, _model_utils, _utils
    from kdmt.mlflow._list_model_versions import _get_model_versions
    from mlflow.exceptions import RestException
except:
    pass



import pandas as pd
import sys

import os, tabulate
from kdmt.mlflow._mlflow_utils import get_mlflow_client
TAG_PARENT_RUN_ID = "mlflow.parentRunId"

def get_best_run(experiment_id_or_name, metric, ascending=False, ignore_nested_runs=False):
    """
    Current search syntax does not allow to check for existence of a tag key so if there are nested runs we have to
    bring all runs back to the client making this costly and unscalable.
    """
    client=get_mlflow_client()
    exp = _mlflow_utils.get_experiment(client, experiment_id_or_name)
    print("Experiment name:",exp.name)

    order_by = "ASC" if ascending else "DESC"
    column = "metrics.{} {}".format(metric, order_by)
    if ignore_nested_runs:
        runs = client.search_runs(exp.experiment_id, "", order_by=[column])
        runs = [ run for run in runs if TAG_PARENT_RUN_ID not in run.data.tags ]
    else:
        runs = client.search_runs(exp.experiment_id, "", order_by=[column], max_results=1)

    run_dict= dict(runs[0].info)
    run_dict["metrics"]=runs[0].data.metrics
    return run_dict
def delete_model(model):
    client=get_mlflow_client()
    _model_utils.delete_model(client, model)


def delete_model_stages(model, stages):
    client=get_mlflow_client()
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
    client=get_mlflow_client()
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

def get_stage_version(model_name, stage='Production'):
    client=get_mlflow_client()
    model=client.get_registered_model(model_name)
    for model_version in model.latest_versions:
        if model_version.current_stage==stage:
            model_version_dict= dict(model_version)
            model_run=client.get_run(model_version.run_id)
            model_version_dict['metrics']=model_run.data.metrics
            model_version_dict["artifact_uri"]=model_run.info.artifact_uri
            return model_version_dict

    return None


def download_model(run, artifact_path, output_dir):
    """
    Downloads the model associated with the model URI.
    - For model scheme, downloads the model associated with the stage/version.
    - For run scheme, downloads the model associated with the run ID and artifact.
    :param: model_uri - MLflow model URI.
    :param:output_dir - Output directory for downloaded model.
    :return: The local path to the downloaded model.
    """

    return mlflow.artifacts.download_artifacts(run['artifact_uri']+"/"+artifact_path, dst_path=output_dir)


def is_model_registered_by_name(model_name):
    client=get_mlflow_client()
    return model_name in  [m.name for m in client.search_registered_models()]
def get_model_registered_by_run_id(model_run_id):
    client=get_mlflow_client()
    runs= [v for mv in [m.latest_versions for m  in client.search_registered_models()] for v in mv if v.run_id == model_run_id]
    if len([v for mv in [m.latest_versions for m  in client.search_registered_models()] for v in mv if v.run_id==model_run_id])>0:
        run=client.get_run(model_run_id)
        run_dict = dict(runs[0])
        run_dict["metrics"]=run.data.metrics
        return run_dict
    else:
        return {}


def find_artifacts(run_id, path, target, max_level=sys.maxsize):
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    return _find_artifacts(run_id, path, target, max_level, 0, [])

def _find_artifacts(run_id, path, target, max_level, level, matches):
    if level+1 > max_level:
        return matches
    client=get_mlflow_client()
    artifacts = client.list_artifacts(run_id,path)
    for art in artifacts:
        #print(f"art_path: {art.path}")
        filename = os.path.basename(art.path)
        if filename == target:
            matches.append(art.path)
        if art.is_dir:
            _find_artifacts(run_id, art.path, target, max_level, level+1, matches)
    return matches


def register_model(registered_model, run, model_artifact, stage="Production"):
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(registered_model)
        print(f"Created new model '{registered_model}'")
    except RestException as e:
        if not "RESOURCE_ALREADY_EXISTS: Registered Model" in str(e):
            raise e
        print(f"Model '{registered_model}' already exists")

    source = f"{run['artifact_uri']}/{model_artifact}"
    print("Source:",source)

    version = client.create_model_version(registered_model, source, run["run_id"])
    print("Version:",version)

    if stage:
        client.transition_model_version_stage(registered_model, version.version, stage)


if __name__ == "__main__":
    import mlflow
    import os


    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"


    model=is_model_registered_by_name("email_signature")
    if model:
        prduction=get_stage_version("email_signature", "Staging")
        download_model(prduction, "current", '.')
    staged=get_stage_version("email_signature", stage='Staging')
    best_run=get_best_run("email_signature", "F1")



    print(best_run)
#    get_best_run("email_signature",
#    "F1")
#    for rm in client.search_registered_models():
#       print(dict(rm))