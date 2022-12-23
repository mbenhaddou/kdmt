"""
List all versions of all registered models with emphasis on if the version's backing run exists.
"""
import os

import click
import pandas as pd
try:
    import mlflow
except:
    pass

from tabulate import tabulate

def list_view(client, models, get_latest):
    data = []
    for model in models:
        if get_latest:
            versions = client.get_latest_versions(model.name)
        else:
            versions = client.search_model_versions(f"name='{model.name}'")
        vdata = []
        for  vr in versions:
            try:
                run = client.get_run(vr.run_id)
                run_stage = run.info.lifecycle_stage
                run_exists = True
            except mlflow.exceptions.RestException:
                run_exists = False
                run_stage = None
            vdata.append([model.name, vr.version, vr.current_stage, dt(vr.creation_timestamp), vr.run_id, run_stage, run_exists ])
        data = data + vdata
    columns = ["Model","Version","Stage", "Creation", "Run ID", "Run stage", "Run exists"]
    df = pd.DataFrame(data, columns = columns)
    which = "Latest" if get_latest else "All"
    print(f"\n{which} {len(data)} versions")
    print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))


def dt(ms):
    from datetime import datetime
    dt = datetime.utcfromtimestamp(ms/1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _get_model_versions(client, model, view, max_results):
    if model == "all":
        models = client.search_registered_models(max_results=max_results)
    else:
        models = [ client.get_registered_model(model) ]
    if view in ["latest","both"]:
        list_view(client, models, True)
    if view in ["all","both"]:
        list_view(client, models, False)
    if view not in ["latest", "all","both"]:
        print(f"ERROR: Bad 'view' value '{view}'")



