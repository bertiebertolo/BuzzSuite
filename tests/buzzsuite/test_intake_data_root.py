"""The Intake tab's data-folder Browse: an existing data folder is used as it is, and a folder inside an
experiment (a lab PC got a whole new data tree inside an experiment's tracking_moving/) is flagged."""
import os
import shutil
import tempfile

from intake_wizard import data_root_choice


def _tree(*dirs, files=()):
    root = tempfile.mkdtemp()
    for d in dirs:
        os.makedirs(os.path.join(root, d))
    for f in files:
        open(os.path.join(root, f), "w").close()
    return root


def test_a_drive_or_top_folder_gets_a_buzzwatch_folder():
    root = _tree("Lab")
    try:
        target, problem = data_root_choice(os.path.join(root, "Lab"))
        assert target == os.path.join(root, "Lab", "Buzzwatch") and problem is None, (target, problem)
    finally:
        shutil.rmtree(root)


def test_an_existing_data_folder_is_used_as_it_is():
    root = _tree("Data/Buzzwatch/Recording", "Data/Buzzwatch/Analysis")
    try:
        picked = os.path.join(root, "Data", "Buzzwatch").replace("\\", "/")  # as the Tk dialog returns it
        target, problem = data_root_choice(picked)
        assert target == os.path.join(root, "Data", "Buzzwatch") and problem is None, (target, problem)
    finally:
        shutil.rmtree(root)


def test_a_folder_inside_an_experiment_is_flagged():
    exp = "Buzzwatch/Analysis/Test/Penzance/ExpA"
    root = _tree(exp + "/tracking_moving", exp + "/final_tracking_data", files=[exp + "/experiment_a.json"])
    try:
        target, problem = data_root_choice(os.path.join(root, exp, "tracking_moving"))
        assert problem and "inside the experiment folder" in problem, problem
        assert problem.endswith(os.path.normpath(os.path.join(root, exp))), problem
        _, problem = data_root_choice(os.path.join(root, exp))
        assert problem and problem.startswith("the experiment folder"), problem
    finally:
        shutil.rmtree(root)


def test_a_folder_inside_an_analysis_tree_is_flagged():
    root = _tree("Buzzwatch/Analysis/Test/Penzance")
    try:
        _, problem = data_root_choice(os.path.join(root, "Buzzwatch", "Analysis", "Test", "Penzance"))
        assert problem and "existing data folder" in problem, problem
    finally:
        shutil.rmtree(root)
