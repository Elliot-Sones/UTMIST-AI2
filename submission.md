Make sure that user/my_agent.py defines SubmittedAgent(Agent) and runs locally
Ensure that pytest -s user/validate.py works and finishes < 60s


In order to accept and qualify your submission, one must first validate their submission by running the validation pipeline in Github Actions. Validation serves as a functionality check for your agent, and is completed done by running your agent against a dummy agent to check that your agent can in fact successful complete a match. A submission is not accepted unless there is a successful validation pipeline recorded. We do not recommend doing debugging through Github Actions as it wastes compute and memory resources. Please test your agent thoroughly locally before launching a validation pipeline run. If our team detects unreasonable and suspicious amount of pipeline runs, we're going to have a serious talk with your team.

File: .github/workflows/agent_validation_pipeline.yaml
Trigger: Manual workflow_dispatch with input username (your GitHub username that owns the fork)
What it does:
Clones your fork https://github.com/<username>/UTMIST-AI2.git on branch main
Runs ./install.sh
Executes pytest -s user/validate.py inside the submission/ folder
Uploads artifacts from submission/results/ (if present)
Timeouts: The validation test has a 60-second timeout.
How to run it:

Go to the UTMIST-AI2 repository’s Actions tab → "RL Tournament Validation Pipeline" → "Run workflow"
Choose main as the branch to run and enter your GitHub username (fork owner username) and start the run
Your submission will be counted as validated if it successfully completes the dummy match in the run.
How to see results:

Open the run → check the live logs for messages like:
"Warming up your agent …"
"Validation match has started …"
"Validation match has completed successfully!"
If your code writes files under submission/results/, they will be available as a downloadable artifact named results-<username>.
Notes:

The default user/validate.py writes the video to validate.mp4 in the working directory.
To persist outputs in CI, write copies into results/, e.g., submission/results/validate.mp4.


Your Agent: Minimum Requirements
Implement SubmittedAgent(Agent) in user/my_agent.py.
The class must be importable without extra interaction and must follow the Agent interface.
Please make sure your weights are publicly downloadable (e.g., gdown with a public link).