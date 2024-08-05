import subprocess

import fire
import pandas as pd
from pplkit.data.interface import DataInterface
from tqdm.auto import tqdm

DATA = "path/to/data"
SCRIPT = "path/to/script"
SAMPLE_SIZE = 10_000


def submit(outcomes: list[str] | None = None) -> None:
    dataif = DataInterface()
    if outcomes is None:
        columns = dataif.load(DATA, index_col=0, nrows=0).to_list()
        outcomes = [
            col.removeprefix("lnRr_")
            for col in columns
            if col.startswith("lnRr_")
            and not col.endswith(("_upper", "_lower"))
        ]

    jobs = pd.DataFrame(dict(outcome=outcomes, sample_size=SAMPLE_SIZE))
    jobs["job_id"] = -1
    jobs["status"] = "UNKNOWN"

    for i, row in tqdm(jobs.query("job_id.isin([-1, ''])").iterrows()):
        outcome, sample_size = row["outcome"], row["sample_size"]
        cmd = f"sbatch {SCRIPT} {outcome} {sample_size}"
        process = subprocess.run(cmd, shell=True, capture_output=True)
        jobs.loc[i, "job_id"] = _get_job_id(process.stdout)
        jobs.loc[i, "status"] = "UNKNOWN"

    dataif.dump(jobs, "jobs.csv")


def inspect() -> None:
    dataif = DataInterface()
    jobs = dataif.load("jobs.csv")
    for i, row in jobs.query(
        "job_id != -1 and status.isin(['UNKNOWN', 'PENDING', 'RUNNING'])"
    ).iterrows():
        job_id = row["job_id"]
        cmd = f"sacct -j {job_id} --format=state --noheader"
        process = subprocess.run(cmd, shell=True, capture_output=True)
        jobs.loc[i, "status"] = _get_status(process.stdout)
    dataif.dump(jobs, "jobs.csv")
    print(
        jobs.query("job_id != -1")
        .groupby("status")
        .size()
        .rename("count")
        .reset_index()
    )


def resubmit() -> None:
    dataif = DataInterface()
    jobs = dataif.load("jobs.csv")
    for i, row in jobs.query("status.isin(['FAILED', 'TIMEOUT'])").iterrows():
        outcome, sample_size = row["outcome"], row["sample_size"]
        cmd = f"sbatch {outcome} {sample_size}"
        process = subprocess.run(cmd, shell=True, capture_output=True)
        jobs.loc[i, "job_id"] = _get_job_id(process.stdout)
        jobs.loc[i, "status"] = "UNKNOWN"
    dataif.dump(jobs, "jobs.csv")


def _get_job_id(stdout: str) -> int:
    return int(stdout.decode("utf-8").strip().split(" ")[-1])


def _get_status(stdout: str) -> str:
    return stdout.decode("utf-8").strip().replace("\n ", "").split(" ")[0]


if __name__ == "__main__":
    fire.Fire({"submit": submit, "inspect": inspect, "resubmit": resubmit})
