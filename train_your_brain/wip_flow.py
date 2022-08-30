import os

from prefect import task, Flow
from train_your_brain.data import get_show_url, get_last_diffusions, save_diffusion_to_history

@task
def get_data(show_id, number_diffusions, history_path):
    """Get newer data"""
    show_url = get_show_url(show_id)
    diffusions_df = get_last_diffusions(show_url, number_diffusions)
    save_diffusion_to_history(diffusions_df, history_path)


def build_flow(show_id, number_diffusions, history_path):
    """Build the prefect workflow for the `taxifare` package"""

    flow_name = os.environ.get("PREFECT_FLOW_NAME")

    with Flow(flow_name) as flow:
        get_data(show_id, number_diffusions, history_path)

    return flow
