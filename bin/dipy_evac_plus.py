#!python

from dipy.workflows.flow_runner import run_flow
from dipy.workflows.segment import EVACPlusFlow

if __name__ == "__main__":
    run_flow(EVACPlusFlow())